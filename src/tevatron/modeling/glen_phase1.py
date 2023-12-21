import os
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict, Optional
from transformers import AutoModelWithLMHead, AutoTokenizer

from tevatron.arguments import (
    GLENP1ModelArguments as ModelArguments,
    GLENP1TrainingArguments as TrainingArguments,
)
from tevatron.tree import TreeBuilder, dec_2d
from tevatron.main_utils import assert_all_frozen

logger = logging.getLogger(__name__)


class GLENP1Model(nn.Module):
    def __init__(self, hf_model, tokenizer: Optional[AutoTokenizer] = None):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.gen_r1_list = []

    def forward(
        self,
        input_ids: torch.Tensor,
        aug_input_ids: torch.Tensor = None,
        encoder_outputs: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        aug_attention_mask: torch.Tensor = None,
        logit_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
        lm_labels: torch.Tensor = None,
        input_mask: torch.Tensor = None,
    ):
        """Forward pass for GLENP1Model"""
        input_mask = None
        if self.model_args.Rdrop > 0 and self.training:
            if aug_input_ids is not None and self.training:
                input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
            elif self.training:
                input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.clone()], dim=0
                )
            if self.model_args.input_dropout and np.random.rand() < 0.5:
                if input_mask is None:
                    input_mask = (
                        torch.rand(input_ids.shape, device=input_ids.device) < 0.9
                    )
                input_ids = torch.where(
                    input_mask == True, input_ids, torch.zeros_like(input_ids)
                )
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask], dim=0
                )
            if lm_labels is not None:
                lm_labels = torch.cat([lm_labels, lm_labels], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, decoder_input_ids], dim=0
                )

        if self.model_args.decoder_input == "doc_rep":
            past_key_values, encoder_outputs = None, None
            decoder_inputs_embeds = self.hf_model.get_input_embeddings()(
                torch.tensor([0], dtype=torch.long, device=torch.device("cuda"))
            )  # [1, H]
            decoder_inputs_embeds = decoder_inputs_embeds.unsqueeze(0).repeat(
                input_ids.shape[0], 1, 1
            )  # [B, 1, H]
            lm_logits = []
            sequence_output = []

            max_output_len_in_batch = (lm_labels != -100).sum(dim=-1).max().item()

            for i in range(max_output_len_in_batch):
                decoder_attention_mask_cur = decoder_attention_mask[:, : i + 1]
                cur_out = self.hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_attention_mask=decoder_attention_mask_cur,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
                if encoder_outputs is None:
                    encoder_outputs = cur_out.encoder_outputs
                past_key_values = cur_out.past_key_values
                decoder_inputs_embeds = cur_out.decoder_hidden_states[-1][
                    :, -1:, :
                ]  # [B, 1, H]
                lm_logits.append(cur_out.logits[:, -1, :])
                sequence_output.append(cur_out.decoder_hidden_states[-1][:, -1:, :])

            lm_logits = torch.stack(lm_logits, dim=1)  # [B, L, V]
            sequence_output = torch.cat(sequence_output, dim=1)  # [B, L, H]
            sequence_output = sequence_output * (
                self.hf_model.model_dim**-0.5
            )  # [B, L, H]

            out = cur_out
            out.logits = lm_logits
            out.lm_logits = lm_logits

            # calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_labels = lm_labels[:, :max_output_len_in_batch]
            orig_loss = loss_fct(
                out.logits.view(-1, self.hf_model.config.vocab_size),
                lm_labels.reshape(-1),
            )
            if self.model_args.Rdrop > 0 and self.training:
                bz = lm_logits.shape[0]
                sl = lm_logits.shape[1]

                # Rdrop contrastive loss
                neg_logits_1 = sequence_output.transpose(0, 1)  # [L, B, V]
                neg_logits_2 = neg_logits_1.transpose(1, 2)  # [L, V, B]
                neg_logits = torch.bmm(neg_logits_1, neg_logits_2)  # [S, B, bz_logits]
                neg_mask = -1e9 * torch.eye(bz).to(neg_logits.device)
                neg_logits = neg_logits + neg_mask.unsqueeze(0)
                neg_logits = F.softmax(
                    neg_logits.view(-1, bz), dim=-1
                )  # [L*B, bz_logits]
                contrast_labels = torch.cat(
                    [torch.arange(bz // 2, bz), torch.arange(0, bz // 2)], dim=-1
                )
                contrast_labels = contrast_labels.to(neg_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(sl, 1).view(-1)
                dist_loss = loss_fct(neg_logits, contrast_labels)

                loss = orig_loss + self.model_args.Rdrop * dist_loss
                out.orig_loss = orig_loss
                out.dist_loss = dist_loss
            else:
                loss = orig_loss
            out.loss = loss

            return out

        else:
            out = self.hf_model(
                input_ids,
                input_mask=input_mask,
                logit_mask=logit_mask,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                lm_labels=lm_labels,
                return_dict=True,
                output_hidden_states=True,
            )
            return out

    def evaluation_step(self, batch: Dict, predefined_id: bool = True):
        """
        Evaluate the model on a batch when validation during training.
        When inference after training, predefined_id is False.
        When validation during training, predefined_id is True.

        Args:
            batch (Dict):
                batch data (source_ids, source_mask, aug_source_ids, aug_source_mask, target_ids, target_mask, rank, oldid)
            predefined_id (bool):
                whether ground truth query exists (True: validation during training, False: inference after training)
                If decoder_input doc_rep, logits are used for breaking ties

        Returns:
            texts (List[str]):
                list of source texts
            preds (List[str]):
                list of predicted query
            target_docids (List[str]):
                list of target docids
            ranks (List[str]):
                list of ranks
        """

        assert (
            self.model_args.gen_method == "greedy"
            if self.model_args.decoder_input == "doc_rep"
            else True
        )

        target_mask = (
            batch["target_mask"].cuda()
            if predefined_id
            else torch.ones((1, self.model_args.max_output_length)).cuda()
        )

        # If ground truth id exists, additional digit is appended to the end of the id
        max_output_length = (
            self.model_args.max_output_length
            if predefined_id
            else self.model_args.max_output_length - 1
        )

        num_return_sequences = self.model_args.num_return_sequences

        # decoder input: doc_rep (d_t)
        if self.model_args.decoder_input == "doc_rep":
            past_key_values, encoder_outputs = None, None
            K = num_return_sequences * 2
            decoder_inputs_embeds = self.hf_model.get_input_embeddings()(
                torch.tensor([0], dtype=torch.long, device=torch.device("cuda"))
            )  # [1, H]
            decoder_inputs_embeds = decoder_inputs_embeds.unsqueeze(0).repeat(
                batch["source_ids"].shape[0], 1, 1
            )  # [B, 1, H]
            decoder_attention_mask_full = torch.ones(
                batch["source_ids"].shape[0],
                max_output_length,
                dtype=torch.long,
                device=torch.device("cuda"),
            )
            decode_tree = self.root if self.model_args.tree else None

            if decode_tree is None:
                raise NotImplementedError("self.tree should be True")

            outs, all_next_token_logits, out_logits = [], [], []
            for i in range(max_output_length):
                decoder_attention_mask = decoder_attention_mask_full[:, : i + 1]
                psg_out = self.hf_model(
                    input_ids=batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                    encoder_outputs=encoder_outputs,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                if encoder_outputs is None:
                    encoder_outputs = psg_out.encoder_outputs
                past_key_values = psg_out.past_key_values
                decoder_inputs_embeds = psg_out.decoder_hidden_states[-1][:, -1:, :]

                if (
                    not predefined_id
                    and self.model_args.mask_special_tokens_for_decoding
                ):
                    psg_out.logits[:, :, self.model_args.special_token_ids] = -float(
                        "inf"
                    )

                next_token_logits = psg_out.logits[:, -1, :]  # [B, V]
                all_next_token_logits.append(
                    next_token_logits.detach().cpu().unsqueeze(1)
                )  # [B, 1, V]

                batch_size, vocab_size = next_token_logits.shape
                scores = F.log_softmax(next_token_logits, dim=-1)  # [B, V]
                if i == 0:
                    mask = torch.ones_like(scores) * -float("inf")  # [B, V]
                    candidates = list(decode_tree.children.keys())
                    mask[:, candidates] = 0  # [B, V]
                    scores += mask  # [B, V]

                    out_prob, out_index = scores.topk(
                        K, dim=-1
                    )  # WARNING top k is doubled
                    cur_prob, cur_index = out_prob, out_index
                    all_index = cur_index.unsqueeze(-1)  # [B, K, 1]
                else:
                    out_prob = scores
                    all_prob = cur_prob[:, :, None] + out_prob.unsqueeze(1)  # [B, K, V]
                    mask = torch.ones_like(all_prob) * -float("inf")

                    for b in range(batch_size):
                        for k in range(K):
                            previous_index = all_index[b, k, :].tolist()
                            cur = decode_tree
                            for value in previous_index:
                                if value not in cur.children:
                                    next_candidates = [1]  # eos token
                                    break
                                else:
                                    cur = cur.children[value]
                            else:
                                next_candidates = list(cur.children.keys())
                            mask[b, k, next_candidates] = 0
                    all_prob += mask

                    top_prob, top_index = all_prob.view(batch_size, -1).topk(
                        K, dim=-1
                    )  # [B, num_return_sequences]
                    before_index = torch.div(
                        top_index, vocab_size, rounding_mode="floor"
                    )
                    after_index = top_index % vocab_size

                    before_index = all_index.gather(
                        1, before_index.unsqueeze(-1).repeat(1, 1, all_index.shape[-1])
                    )
                    all_index = torch.cat(
                        [before_index, after_index.unsqueeze(-1)], dim=-1
                    )
                    cur_prob = top_prob

            all_index = all_index[:, :num_return_sequences, :]  # [B, K//2, L]
            outs = all_index.reshape(
                -1, max_output_length
            )  # [B*num_return_sequences, L]
            outs = outs.detach().cpu()

            if predefined_id:
                dec = []
                for out in outs:
                    if self.tokenizer.eos_token_id in out:
                        out = out[: list(out).index(self.tokenizer.eos_token_id)]
                    dec.append(
                        "-".join(
                            self.tokenizer.convert_ids_to_tokens(
                                out, skip_special_tokens=True
                            )
                        )
                    )
            else:
                # Calculate logits for breaking ties
                out_logits = torch.zeros_like(
                    outs, dtype=torch.float32
                )  # [B*num_return_sequences, L]
                all_next_token_logits = (
                    torch.cat(all_next_token_logits, dim=1).detach().cpu()
                )  # [B, L, V]
                for b in range(batch_size):
                    outs_b = outs[
                        b * num_return_sequences : (b + 1) * num_return_sequences
                    ]  # [num_return_sequences, L]
                    all_next_token_logits_b = all_next_token_logits[b : b + 1].repeat(
                        num_return_sequences, 1, 1
                    )  # [num_return_sequences, L, V]
                    out_logits_b = all_next_token_logits_b.gather(
                        2, outs_b.unsqueeze(-1)
                    ).squeeze(
                        -1
                    )  # [num_return_sequences, L]
                    out_logits[
                        b * num_return_sequences : (b + 1) * num_return_sequences
                    ] = out_logits_b

                dec, temp_dec = [], []
                for out_cnt, ids in enumerate(outs):
                    pred_id = "<->".join(
                        self.tokenizer.convert_ids_to_tokens(
                            ids, skip_special_tokens=False
                        )
                    )
                    out_logits_i = out_logits[out_cnt]

                    if self.docid2num_docs[pred_id] == 1:
                        oldid = self.docid2oldids[pred_id][0]
                        temp_dec.append(oldid + "<->" + pred_id)
                    else:
                        if self.model_args.reranking == "random":
                            cand_oldids = self.docid2oldids[pred_id]  # list of oldids
                            cand_oldids = np.random.permutation(cand_oldids)
                            pred_id_list = [
                                oldid + "<->" + pred_id for oldid in cand_oldids
                            ]
                            temp_dec += pred_id_list
                        elif self.model_args.reranking in ["cosine", "mse"]:
                            cand_oldids = self.docid2oldids[pred_id]  # list of oldids
                            rank_scores = []
                            for cand_oldid in cand_oldids:
                                docid_logit = self.oldid2docid_logit[cand_oldid]  # [L]
                                if self.model_args.reranking == "cosine":
                                    rank_score = torch.cosine_similarity(
                                        docid_logit.unsqueeze(0),
                                        out_logits_i.unsqueeze(0),
                                    ).item()
                                elif self.model_args.reranking == "mse":
                                    rank_score = (
                                        (docid_logit - out_logits_i)
                                        .pow(2)
                                        .mean()
                                        .item()
                                    )
                                rank_scores.append(rank_score)
                            rank_scores = np.array(rank_scores)
                            order_index = np.argsort(-rank_scores)
                            ordered_cand_oldids = np.array(cand_oldids)[order_index]
                            pred_id_list = [
                                oldid + "<->" + pred_id for oldid in ordered_cand_oldids
                            ]
                            temp_dec += pred_id_list
                        else:
                            raise NotImplementedError(
                                "reranking method should be one of ['random', 'cosine', 'mse']"
                            )

                    if out_cnt % num_return_sequences == num_return_sequences - 1:
                        temp_dec = temp_dec[:num_return_sequences]
                        dec += temp_dec
                        temp_dec = []

                targets = []
                for oldid in batch["oldid"]:
                    targets.append(oldid + "<->" + self.oldid2docid[oldid])
                batch["rank"][0][0] = targets

        # decoder input: doc_id (z_t)
        else:
            outs = self.hf_model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=False,
                decoder_attention_mask=target_mask,
                max_length=max_output_length,
                num_beams=num_return_sequences,
                length_penalty=self.model_args.length_penalty,
                num_return_sequences=num_return_sequences,
                early_stopping=self.model_args.early_stopping,
                vocab_size=self.config.decode_vocab_size,
                decode_tree=self.root,
            )

            if predefined_id:
                dec = [
                    "-".join(
                        self.tokenizer.convert_ids_to_tokens(
                            ids, skip_special_tokens=True
                        )
                    )
                    for ids in outs
                ]
            else:
                outs = outs[:, 1:]
                dec = []
                temp_dec = []
                for out_cnt, ids in enumerate(outs):
                    pred_id = "<->".join(
                        self.tokenizer.convert_ids_to_tokens(
                            ids, skip_special_tokens=False
                        )
                    )
                    if self.docid2num_docs[pred_id] == 1:
                        oldid = self.docid2oldids[pred_id][0]
                        temp_dec.append(oldid + "<->" + pred_id)
                    else:
                        cand_oldids = self.docid2oldids[pred_id]  # list of oldids
                        cand_oldids = np.random.permutation(cand_oldids)
                        pred_id_list = [
                            oldid + "<->" + pred_id for oldid in cand_oldids
                        ]
                        temp_dec += pred_id_list
                    if out_cnt % num_return_sequences == num_return_sequences - 1:
                        temp_dec = temp_dec[:num_return_sequences]
                        dec += temp_dec
                        temp_dec = []

                targets = []
                for oldid in batch["oldid"]:
                    targets.append(oldid + "<->" + self.oldid2docid[oldid])
                batch["rank"][0][0] = targets

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["source_ids"]
        ]
        preds = dec_2d(dec, num_return_sequences)
        target_docids = batch["rank"][0][0]
        ranks = [str(a.item()) for a in batch["rank"][0][1]]

        return texts, preds, target_docids, ranks

    def build_tree(self):
        """Build tree for decoding while validation during training"""
        data_args = self.trainer.data_args
        builder = TreeBuilder()

        # load oldid2newid file
        if data_args.dataset_name == "nq320k":
            newid_df = pd.read_csv(
                f"data/nq320k/ID_NQ_{data_args.id_class}.tsv",
                sep="\t",
                dtype=str,
            )
            if data_args.small_set in [1, 10]:
                small_id_df = pd.read_csv(
                    f"data/nq320k/SMALL_NQ_{data_args.small_set}_oldid.tsv",
                    sep="\t",
                    dtype=str,
                )
                newid_df = newid_df[newid_df["oldid"].isin(small_id_df["oldid"])]
        elif data_args.dataset_name == "marco_passage":
            newid_df = pd.read_csv(
                f"data/marco_passage/ID_MARCO_{data_args.id_class}.tsv",
                sep="\t",
                dtype=str,
            )
            if data_args.small_set in [1]:
                small_id_df = pd.read_csv(
                    f"data/marco_passage/SMALL_MARCO_{data_args.small_set}_oldid.tsv",
                    sep="\t",
                    dtype=str,
                )
                newid_df = newid_df[newid_df["oldid"].isin(small_id_df["oldid"])]
        oldid2newid = dict(zip(newid_df["oldid"], newid_df[data_args.id_class]))

        # Load all ids
        if data_args.dataset_name == "nq320k":
            train_filename = "data/nq320k/GTQ_NQ_train.tsv"
            dev_filename = "data/nq320k/GTQ_NQ_dev.tsv"
            df_train = pd.read_csv(
                train_filename, encoding="utf-8", sep="\t", dtype=str
            )
            df_dev = pd.read_csv(dev_filename, encoding="utf-8", sep="\t", dtype=str)
            # insert newid and remove oldid
            df_train.insert(1, data_args.id_class, df_train["oldid"].map(oldid2newid))
            df_train = df_train.loc[:, ["query", data_args.id_class]]
            df_dev.insert(1, data_args.id_class, df_dev["oldid"].map(oldid2newid))
            df_dev = df_dev.loc[:, ["query", data_args.id_class]]
            df = pd.merge(df_train, df_dev, how="outer")
        elif data_args.dataset_name == "marco_passage":
            file_name = f"data/marco_passage/ID_MARCO_{data_args.id_class}.tsv"
            df = pd.read_csv(file_name, encoding="utf-8", sep="\t", dtype=str)
        elif data_args.dataset_name in ["nfcorpus", "arguana", "scidocs"]:
            fname = f"data/BEIR_dataset/{data_args.dataset_name}/corpus_newid.tsv"
            df = pd.read_csv(
                fname, encoding="utf-8", sep="\t", dtype={data_args.id_class: str}
            ).loc[:, [data_args.id_class]]

        # Build tree
        if self.model_args.tree == 1:
            newid_list = df[data_args.id_class].tolist()
            for newid in tqdm(newid_list, desc="Building Tree", dynamic_ncols=True):
                toks = str(newid).split(
                    "-"
                )  # marketing-email-emails-mail -> ['marketing', 'email', 'emails', 'mail']
                toks = self.tokenizer.convert_tokens_to_ids(toks) + [
                    1
                ]  # ['marketing', 'email', 'emails', 'mail'] -> [5821, 10373, 22028, 5653, 1]
                builder.add(toks)
            self.root = builder.build()
        else:
            self.root = None

        self.docid2num_docs, self.docid2oldids, self.oldid2docid_logit = (
            None,
            None,
            None,
        )
        self.oldid2docid = oldid2newid

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        tokenizer: Optional[AutoTokenizer] = None,
        **hf_kwargs,  # config, cache_dir
    ):
        """Build model from pretrained model or local checkpoint"""
        cls.config = hf_kwargs["config"]
        cls.model_args = model_args

        # import t5 backbone model here due to ImportError (circle import)
        from tevatron.modeling import T5ForConditionalGeneration_GLEN

        if os.path.isdir(model_args.model_name_or_path):
            logger.info(
                f"loading model weight from local {model_args.model_name_or_path}"
            )
            hf_model = T5ForConditionalGeneration_GLEN(cls.config)
        else:
            logger.info(
                f"loading model weight from huggingface {model_args.model_name_or_path}"
            )
            hf_model = T5ForConditionalGeneration_GLEN(cls.config)

        if model_args.pretrain_encoder:
            pretrain_model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path
            )
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in hf_model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
            print("Load Pretrain Encoder !")

        if model_args.pretrain_decoder:
            pretrain_model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path
            )
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in hf_model.named_parameters():
                if name.startswith(("decoder.")):
                    if pretrain_params.get(name) is not None:
                        with torch.no_grad():
                            param.copy_(pretrain_params[name])
                    else:
                        print(f"{name} is not in pretranied model")
            print("Load Pretrain Decoder !")

        if model_args.freeze_embeds:
            cls.freeze_embeds()
        if model_args.freeze_encoder:
            cls.freeze_params(hf_model.get_encoder())
            assert_all_frozen(hf_model.get_encoder())

        model = cls(hf_model=hf_model, tokenizer=tokenizer)
        return model

    @classmethod
    def load(
        cls,
        model_args: ModelArguments,
        tokenizer: Optional[AutoTokenizer] = None,
        **hf_kwargs,  # config, cache_dir
    ):
        """Load model from pretrained model or local checkpoint"""
        cls.config = hf_kwargs["config"]
        cls.model_args = model_args

        # import t5 backbone model here due to ImportError (circle import)
        from tevatron.modeling import T5ForConditionalGeneration_GLEN

        if os.path.isdir(model_args.model_name_or_path):
            logger.info(
                f"loading model weight from local {model_args.model_name_or_path}"
            )
            hf_model = T5ForConditionalGeneration_GLEN(cls.config)
        else:
            logger.info(
                f"loading model weight from huggingface {model_args.model_name_or_path}"
            )
            hf_model = T5ForConditionalGeneration_GLEN(cls.config)

        if model_args.freeze_embeds:
            cls.freeze_embeds()
        if model_args.freeze_encoder:
            cls.freeze_params(hf_model.get_encoder())
            assert_all_frozen(hf_model.get_encoder())

        model = cls(hf_model=hf_model, tokenizer=tokenizer)
        return model

    def save(self, output_dir: str):
        """Save model to output_dir"""
        self.hf_model.save_pretrained(os.path.join(output_dir))

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.hf_model.shared)
            for d in [self.hf_model.encoder, self.hf_model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.hf_model.shared)
            for d in [self.hf_model.encoder, self.hf_model.decoder]:
                self.freeze_params(d.embed_tokens)

    def freeze_params(self, model: nn.Module):
        """Freeze parameters of model"""
        for par in model.parameters():
            par.requires_grad = False

    def make_doc_id(self, batch: Dict):
        """
        Make doc id for validation during training

        Args:
            batch (Dict):
                batch data (source_ids, source_mask, aug_source_ids, aug_source_mask, target_ids, target_mask, rank, oldid)
        Returns:
            oldids (List[str]):
                list of oldids
            texts (List[str]):
                list of source texts
            preds (List[str]):
                list of predicted query
            out_logits (List[str]):
                list of logits for breaking ties
        """

        with torch.no_grad():
            past_key_values, encoder_outputs = None, None
            decoder_inputs_embeds = self.lm_p.get_input_embeddings()(
                torch.tensor([0], dtype=torch.long, device=torch.device("cuda"))
            )  # [1, H]
            decoder_inputs_embeds = decoder_inputs_embeds.unsqueeze(0).repeat(
                batch["source_ids"].shape[0], 1, 1
            )  # [B, 1, H]
            decoder_attention_mask_full = torch.ones(
                batch["source_ids"].shape[0],
                self.model_args.max_output_length - 1,
                dtype=torch.long,
                device=torch.device("cuda"),
            )
            outs, out_logits = [], []
            for i in range(self.model_args.max_output_length - 1):
                decoder_attention_mask = decoder_attention_mask_full[:, : i + 1]
                psg_out = self.lm_p(
                    input_ids=batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                    encoder_outputs=encoder_outputs,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                if encoder_outputs is None:
                    encoder_outputs = psg_out.encoder_hidden_states
                past_key_values = psg_out.past_key_values
                decoder_inputs_embeds = psg_out.decoder_hidden_states[-1][:, -1:, :]
                if self.model_args.mask_special_tokens_for_decoding:
                    psg_out.logits[:, :, self.model_args.special_token_ids] = -float(
                        "inf"
                    )
                out = psg_out.logits[:, -1, :].argmax(dim=-1)  # [B, L-1]
                out_logit = (
                    psg_out.logits[:, -1, :].gather(1, out.unsqueeze(-1)).squeeze(-1)
                )
                outs.append(out.cpu().numpy())
                out_logits.append(out_logit.cpu().detach().numpy())
            outs = np.stack(outs, axis=1)
            out_logits = np.stack(out_logits, axis=1)

            preds = []
            for ids in outs:
                preds.append(
                    "<->".join(
                        self.tokenizer.convert_ids_to_tokens(
                            ids, skip_special_tokens=False
                        )
                    )
                )

        texts = [
            self.tokenizer.decode(ids.numpy(), skip_special_tokens=True)
            for ids in batch["source_ids"]
        ]
        oldids = [oldid for oldid in batch["oldids"]]
        out_logits = np.round(out_logits.astype(np.float64), 4)

        return oldids, texts, preds, out_logits
