import torch
import logging

import numpy as np

from torch.nn import functional as F
from typing import Dict, Optional

from tevatron.tree import dec_2d
from tevatron.modeling import EncoderDecoderModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class GLENP2Model(EncoderDecoderModelForSeq2SeqLM):
    def encode_passage(self, psg: Optional[Dict]):
        """Encode passage"""
        if psg is None:
            return None, None, None

        past_key_values, encoder_outputs = None, None
        decoder_inputs_embeds = self.lm_p.get_input_embeddings()(
            psg["decoder_input_ids"]
        )
        decoder_attention_mask_full = torch.ones(
            psg["input_ids"].shape[0],
            self.model_args.num_multi_vectors,
            dtype=torch.long,
            device=decoder_inputs_embeds.device,
        )

        p_reps = []
        for i in range(self.model_args.num_multi_vectors):
            decoder_attention_mask = decoder_attention_mask_full[:, : i + 1]
            psg_out = self.lm_p(
                input_ids=psg["input_ids"],
                attention_mask=psg["attention_mask"],
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
            p_reps.append(psg_out.decoder_hidden_states[-1][:, -1:, :])
        p_reps = torch.cat(p_reps, dim=1)  # (B * train_n_passages, M, H)

        p_reps = p_reps * (p_reps.size(-1) ** -0.5)
        p_reps_dt = p_reps.clone()

        lm_logits = p_reps @ self.lm_p.shared.weight.T

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
            lm_logits[:, :, special_token_ids] = -float("inf")

        if self.model_args.do_docid_temperature_annealing:
            first_temperature = self.model_args.docid_temperature
            cur_epoch = (
                self.trainer.state.epoch if hasattr(self, "trainer") else self.cur_epoch
            )
            temperature = max(
                self.model_args.docid_temperature_min,
                first_temperature * np.exp(-cur_epoch),
            )
        else:
            temperature = self.model_args.docid_temperature

        lm_logits = lm_logits / temperature
        lm_attention = torch.softmax(lm_logits, dim=-1)
        p_reps = lm_attention @ self.lm_p.shared.weight

        return p_reps, lm_attention, p_reps_dt

    def encode_query(self, qry: Optional[Dict]):
        """Encode query"""
        if qry is None:
            return None

        past_key_values, encoder_outputs = None, None
        decoder_inputs_embeds = self.lm_q.get_input_embeddings()(
            qry["decoder_input_ids"]
        )
        decoder_attention_mask_full = torch.ones(
            qry["input_ids"].shape[0],
            self.model_args.num_multi_vectors,
            dtype=torch.long,
            device=decoder_inputs_embeds.device,
        )

        q_reps = []
        for i in range(self.model_args.num_multi_vectors):
            decoder_attention_mask = decoder_attention_mask_full[:, : i + 1]
            qry_out = self.lm_q(
                input_ids=qry["input_ids"],
                attention_mask=qry["attention_mask"],
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=True,
                encoder_outputs=encoder_outputs,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
            if encoder_outputs is None:
                encoder_outputs = qry_out.encoder_hidden_states
            past_key_values = qry_out.past_key_values
            decoder_inputs_embeds = qry_out.decoder_hidden_states[-1][:, -1:, :]
            q_reps.append(qry_out.decoder_hidden_states[-1][:, -1:, :])
        q_reps = torch.cat(q_reps, dim=1)  # (B * train_n_passages, M, H)
        q_reps = q_reps * (q_reps.size(-1) ** -0.5)
        return q_reps

    def compute_similarity(self, q_reps: torch.Tensor, p_reps: torch.Tensor):
        """Compute similarity between query and passage"""
        # q_reps: (B, M, H), p_reps: (B * train_n_passages, M, H)
        q_reps = q_reps.permute(1, 0, 2)  # (M, B, H)
        p_reps = p_reps.permute(1, 0, 2)  # (M, B * train_n_passages, H)
        scores = torch.bmm(q_reps, p_reps.permute(0, 2, 1)).permute(
            1, 0, 2
        )  # (M, B, B*train_n_passages) -> (B, M, B*train_n_passages)
        scores = scores.sum(dim=1) / self.model_args.num_multi_vectors
        return scores

    def make_doc_id(self, batch: Dict):
        """Process document and make doc_id"""
        self.eval()
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

    def evaluation_step(self, batch: Dict, predefined_id: bool = False):
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
        # TODO Merge code with GLEN P1 evaluation_step

        # If ground truth id exists, additional digit is appended to the end of the id
        max_output_length = (
            self.model_args.max_output_length
            if predefined_id
            else self.model_args.max_output_length - 1
        )

        num_return_sequences = self.model_args.num_return_sequences

        with torch.no_grad():
            past_key_values, encoder_outputs = None, None
            K = num_return_sequences * 2
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
            decode_tree = self.root if self.model_args.tree else None

            outs, out_logits, all_next_token_logits = [], [], []
            for i in range(max_output_length):
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
                    )  ## WARNING top k is doubled
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

        texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in batch["source_ids"]
        ]
        preds = dec_2d(dec, num_return_sequences)
        target_docids = batch["rank"][0][0]
        ranks = [str(a.item()) for a in batch["rank"][0][1]]

        return texts, preds, target_docids, ranks
