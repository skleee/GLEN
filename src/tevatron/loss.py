import torch
import numpy as np

from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F


class SimpleContrastiveLoss:
    def __call__(
        self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = "mean"
    ):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0,
                x.size(0) * target_per_qry,
                target_per_qry,
                device=x.device,
                dtype=torch.long,
            )
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class GLENLoss:
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        target: Tensor = None,
        reduction: str = "mean",
        **kwargs
    ):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0,
                x.size(0) * target_per_qry,
                target_per_qry,
                device=x.device,
                dtype=torch.long,
            )
        model = kwargs["model"]
        tokenizer = model.tokenizer

        q_reps = x
        p_reps_dt = y

        # get representations
        lm_logits_p = p_reps_dt @ model.lm_p.shared.weight.T
        if model.model_args.mask_special_tokens_for_decoding:
            special_token_ids = tokenizer.all_special_ids
            special_token_ids = [
                x
                for x in special_token_ids
                if x
                not in [
                    tokenizer.bos_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ]
            ]
            lm_logits_p[:, :, special_token_ids] = -float("inf")
        if model.model_args.do_docid_temperature_annealing:
            first_temperature = model.model_args.docid_temperature
            cur_epoch = (
                model.trainer.state.epoch
                if hasattr(model, "trainer")
                else model.cur_epoch
            )
            temperature = max(
                model.model_args.docid_temperature_min,
                first_temperature * np.exp(-cur_epoch),
            )
        else:
            temperature = model.model_args.docid_temperature

        lm_logits_p = lm_logits_p / temperature
        lm_attention = torch.softmax(lm_logits_p, dim=-1)
        p_reps = lm_attention @ model.lm_p.shared.weight
        p_attention = lm_attention

        # pairwise loss
        scores = model.compute_similarity(q_reps, p_reps)  # (B, B * train_n_passages)
        scores = scores.view(q_reps.size(0), -1)  # (B, B * train_n_passages)

        scores = scores / model.softmax_temperature

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss = model.compute_loss(scores, target) * model.model_args.infonce_loss

        # pointwise loss
        lm_logits_q = None
        if (
            "q_to_docid_loss" in model.model_args.__dict__
            and model.model_args.q_to_docid_loss > 0
        ):
            lm_logits_q = q_reps @ model.lm_q.shared.weight.T  # (B, L, V)
            if model.model_args.mask_special_tokens_for_decoding:
                special_token_ids = tokenizer.all_special_ids
                special_token_ids = [
                    x
                    for x in special_token_ids
                    if x
                    not in [
                        tokenizer.bos_token_id,
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                    ]
                ]
                lm_logits_q[:, :, special_token_ids] = -1e9

            lm_logits_q = lm_logits_q / model.softmax_temperature  # (B, L, V)

            # For each query, select the corresponding positive document
            pos_doc = torch.arange(q_reps.size(0), dtype=torch.long)  # (B)
            pos_doc = pos_doc * (p_reps.size(0) // q_reps.size(0))  # (B)
            pos_p_attention = p_attention[
                pos_doc, :, :
            ]  # (B * train_n_passages, L, V) -> (B, L, V)

            lm_targets = pos_p_attention.argmax(dim=-1)  # (B, L)
            lm_loss = model.cross_entropy(
                lm_logits_q.view(-1, lm_logits_q.size(-1)), lm_targets.view(-1)
            )

            loss = loss + lm_loss * model.model_args.q_to_docid_loss

        # pointwise loss
        if (
            "cosine_point_loss" in model.model_args.__dict__
            and model.model_args.cosine_point_loss > 0
        ):
            if lm_logits_q is None:
                lm_logits_q = q_reps @ model.lm_q.shared.weight.T  # (B, L, V)
                if model.model_args.mask_special_tokens_for_decoding:
                    special_token_ids = tokenizer.all_special_ids
                    special_token_ids = [
                        x
                        for x in special_token_ids
                        if x
                        not in [
                            tokenizer.bos_token_id,
                            tokenizer.eos_token_id,
                            tokenizer.pad_token_id,
                        ]
                    ]
                    lm_logits_q[:, :, special_token_ids] = -1e9

                lm_logits_q = lm_logits_q / model.softmax_temperature  # (B, L, V)

            pos_doc_lm_logits = (
                p_reps_dt[pos_doc] @ model.lm_p.shared.weight.T
            )  # (B, L, V)
            query_lm_logits = lm_logits_q  # (B, L, V)

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

            loss = loss + cosine_loss * model.model_args.cosine_point_loss

        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert (
            dist.is_initialized()
        ), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
