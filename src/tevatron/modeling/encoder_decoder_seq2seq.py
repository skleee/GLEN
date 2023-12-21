import os
import time
import copy
import logging

import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.file_utils import ModelOutput
from typing import Dict, Optional

from tevatron.tree import TreeBuilder
from tevatron.arguments import (
    GLENP2ModelArguments as ModelArguments,
    GLENP2TrainingArguments as TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class EncoderDecoderOutputForSeq2SeqLM(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderDecoderModelForSeq2SeqLM(nn.Module):
    TRANSFORMER_CLS = AutoModelForSeq2SeqLM

    def __init__(
        self,
        lm_q: PreTrainedModel,
        lm_p: PreTrainedModel,
        untie_encoder: bool = False,
        negatives_x_device: bool = False,
        tokenizer=None,
    ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.tokenizer = tokenizer
        self.gen_r1_list = []

    def forward(
        self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None
    ):
        # query, passage: input_ids (B * L), attention_mask (B * L)
        decoder_start_token_id = self.config.decoder_start_token_id
        if query is not None:
            query["decoder_input_ids"] = (
                torch.zeros(query["input_ids"].shape[0], 1, dtype=torch.long)
                .fill_(decoder_start_token_id)
                .to(query["input_ids"].device)
            )  # (B, 1)
            query["decoder_attention_mask"] = torch.ones(
                query["input_ids"].shape[0], 1, dtype=torch.long
            ).to(
                query["input_ids"].device
            )  # (B, 1)
        if passage is not None:
            passage["decoder_input_ids"] = (
                torch.zeros(passage["input_ids"].shape[0], 1, dtype=torch.long)
                .fill_(decoder_start_token_id)
                .to(passage["input_ids"].device)
            )  # (B * train_n_passages, 1)
            passage["decoder_attention_mask"] = torch.ones(
                passage["input_ids"].shape[0], 1, dtype=torch.long
            ).to(
                passage["input_ids"].device
            )  # (B * train_n_passages, 1)

        q_reps = self.encode_query(query)

        if passage is None:
            p_reps, p_reps_dt = None, None  # for gradcache
        else:
            p_reps, p_attention, p_reps_dt = self.encode_passage(passage)

            predid2oldid = self.trainer.train_dataset.predid2oldid
            oldid2predid = self.trainer.train_dataset.oldid2predid

            pred_docids = p_attention.argmax(dim=-1).cpu().tolist()

            # Cache predicted ids for prefix-aware negative sampling
            if (
                self.trainer.data_args.negative_passage_type == "self"
                and p_reps is not None
            ):
                for oldid, predid in zip(passage["oldid"].cpu().tolist(), pred_docids):
                    for j in range(1, len(predid) + 1):
                        cur_predid = "-".join([str(x) for x in predid[:j]])
                        predid2oldid[cur_predid].add(oldid)
                    oldid2predid[oldid] = cur_predid

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderDecoderOutputForSeq2SeqLM(q_reps=q_reps, p_reps=p_reps_dt)

        # for training
        loss, lm_loss = 0, 0
        gen_r1 = 0
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(
                q_reps, p_reps
            )  # (B, B * train_n_passages)
            scores = scores.view(q_reps.size(0), -1)  # (B, B * train_n_passages)

            scores = scores / self.softmax_temperature

            target = torch.arange(
                scores.size(0), device=scores.device, dtype=torch.long
            )
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores, target)

            if "infonce_loss" in self.model_args.__dict__:
                loss = loss * self.model_args.infonce_loss

            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction

            lm_logits = q_reps @ self.lm_q.shared.weight.T  # (B, L, V)

            if self.model_args.mask_special_tokens_for_decoding:
                special_token_ids = self.tokenizer.all_special_ids
                special_token_ids = [
                    x
                    for x in special_token_ids
                    if x
                    not in [
                        self.tokenizer.bos_token_id,
                        self.tokenizer.eos_token_id,
                        self.tokenizer.pad_token_id,
                    ]
                ]
                lm_logits[:, :, special_token_ids] = -1e9

            lm_logits = lm_logits / self.softmax_temperature  # (B, L, V)

            # For each query, select the corresponding positive document
            pos_doc = torch.arange(q_reps.size(0), dtype=torch.long)  # (B)
            pos_doc = pos_doc * (p_reps.size(0) // q_reps.size(0))  # (B)
            pos_p_attention = p_attention[
                pos_doc, :, :
            ]  # (B * train_n_passages, L, V) -> (B, L, V)

            lm_targets = pos_p_attention.argmax(dim=-1)  # (B, L)
            if (
                "q_to_docid_loss" in self.model_args.__dict__
                and self.model_args.q_to_docid_loss > 0
            ):
                lm_loss = self.cross_entropy(
                    lm_logits.view(-1, lm_logits.size(-1)), lm_targets.view(-1)
                )

                loss += lm_loss * self.model_args.q_to_docid_loss

            if (
                "cosine_point_loss" in self.model_args.__dict__
                and self.model_args.cosine_point_loss > 0
            ):
                pos_doc_lm_logits = (
                    p_reps_dt[pos_doc] @ self.lm_p.shared.weight.T
                )  # (B, L, V)
                query_lm_logits = lm_logits  # (B, L, V)

                pos_doc_id_weight, pos_doc_id = pos_doc_lm_logits.max(dim=-1)  # (B, L)
                query_id_weight = query_lm_logits.gather(
                    dim=-1, index=pos_doc_id.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B, L)

                cosine_sim = F.cosine_similarity(
                    pos_doc_id_weight, query_id_weight, dim=-1
                )  # (B)
                cosine_loss = torch.mean(1 - cosine_sim)  # ()

                loss += cosine_loss * self.model_args.cosine_point_loss

                query_preds = lm_logits.argmax(dim=-1)  # (B, L)
                gen_r1 = (query_preds == lm_targets).float().mean(dim=1) == 1  # (B)
                self.gen_r1_list += gen_r1.cpu().tolist()

        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderDecoderOutputForSeq2SeqLM(
            loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps_dt
        )

    def encode_passage(self, psg):
        raise NotImplementedError("EncoderDecoderModel is an abstract class")

    def encode_query(self, qry):
        raise NotImplementedError("EncoderDecoderModel is an abstract class")

    def compute_similarity(self, q_reps, p_reps):
        raise NotImplementedError("EncoderModel is an abstract class")

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def build_tree(self, log_step: int = None, log_file: str = None):
        docid_df = pd.read_csv(
            self.model_args.docid_file_name,
            sep="\t",
            names=["oldid", "docid", "docid_logit", "text"],
            dtype={"oldid": str, "docid": str, "docid_logit": str, "text": str},
        ).loc[:, ["oldid", "docid", "docid_logit"]]

        lines = []

        # Log the time and step
        time_str = time.strftime("%Y%m%d-%H%M%S")
        lines.append(f"TIME={time_str}")
        lines.append(f"STEP={log_step}")

        # Check the uniqueness of docid and build docid2num_docs
        num_uniques = len(set(docid_df.docid))
        unique_ratio = num_uniques / len(docid_df)
        lines.append(
            f"num_uniques: {num_uniques}/{len(docid_df)} ({unique_ratio*100:.2f}%)"
        )
        lines.append("[Frequent Collision]")
        lines.append(docid_df.docid.value_counts()[:5].to_string())
        num_unique_tokens = [0] * 10
        docid2num_docs = dict(docid_df.docid.value_counts())
        for docid in docid_df.docid.unique():
            tokens = docid.split("<->")
            num_unique_token = len(set(tokens))
            num_unique_tokens[num_unique_token - 1] += 1
        lines.append(f"distribution of number of unique tokens: {num_unique_tokens}")

        # Build docid2oldids, oldid2docid_logit
        docid2oldids = defaultdict(list)
        oldid2docid_logit = dict()
        for docid, oldid, docid_logit in docid_df[
            ["docid", "oldid", "docid_logit"]
        ].values:
            docid2oldids[docid].append(oldid)
            oldid2docid_logit[oldid] = torch.tensor(
                [float(x) for x in docid_logit.split("<->")]
            )

        # Build oldid2docid
        oldid2docid = dict(zip(docid_df.oldid, docid_df.docid))
        if self.model_args.tree == 1:
            builder = TreeBuilder()
            all_id = []
            for docid in list(oldid2docid.values()):
                toks = docid.split("<->")
                toks = self.tokenizer.convert_tokens_to_ids(toks)
                if len(toks) != self.model_args.max_output_length - 1:
                    print(
                        f"length of docid of {docid} is not {self.model_args.max_output_length}"
                    )
                    toks = toks[: self.model_args.max_output_length - 1]
                all_id.append(toks)
                builder.add(toks)
            self.root = builder.build()

        self.docid2num_docs = docid2num_docs
        self.docid2oldids = docid2oldids
        self.oldid2docid_logit = oldid2docid_logit
        self.oldid2docid = oldid2docid

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write("\n".join(lines) + "\n")
        else:
            print("\n".join(lines))

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        tokenizer=None,
        **hf_kwargs,  # config, cache_dir
    ):
        cls.config = hf_kwargs["config"]
        cls.softmax_temperature = model_args.softmax_temperature
        cls.model_args = model_args

        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(
                    model_args.model_name_or_path, "query_model"
                )
                _psg_model_path = os.path.join(
                    model_args.model_name_or_path, "passage_model"
                )
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder,
            tokenizer=tokenizer,
        )
        return model

    @classmethod
    def load(
        cls,
        model_args: ModelArguments,
        tokenizer: Optional[AutoTokenizer] = None,
        **hf_kwargs,  # config, cache_dir
    ):
        cls.config = hf_kwargs["config"]
        cls.softmax_temperature = model_args.softmax_temperature
        cls.model_args = model_args

        # load local
        untie_encoder = True
        if os.path.isdir(model_args.model_name_or_path):
            _qry_model_path = os.path.join(model_args.model_name_or_path, "query_model")
            _psg_model_path = os.path.join(
                model_args.model_name_or_path, "passage_model"
            )
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(
                    f"loading model weight from {model_args.model_name_or_path}"
                )
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    model_args.model_name_or_path, **hf_kwargs
                )
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_args.model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name_or_path, **hf_kwargs
            )
            lm_p = lm_q

        model = cls(lm_q=lm_q, lm_p=lm_p, untie_encoder=untie_encoder)
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, "query_model"))
            os.makedirs(os.path.join(output_dir, "passage_model"))
            self.lm_q.save_pretrained(os.path.join(output_dir, "query_model"))
            self.lm_p.save_pretrained(os.path.join(output_dir, "passage_model"))
        else:
            self.lm_q.save_pretrained(output_dir)
