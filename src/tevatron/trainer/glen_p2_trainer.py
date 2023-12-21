import os
import math
import time
import logging
import pandas as pd
import numpy as np

import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.trainer_pt_utils import find_batch_size
from transformers.file_utils import is_datasets_available

from tevatron.loss import GLENLoss
from tevatron.metrics import compute_recall, compute_mrr
from tevatron.main_utils import split_dense_inputs, get_dense_rep

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

if is_datasets_available():
    import datasets


class GLENP2Trainer(Trainer):
    def __init__(self, eval_dataset_doc: Optional[Dataset] = None, *args, **kwargs):
        super(GLENP2Trainer, self).__init__(*args, **kwargs)
        self.eval_dataset_doc = eval_dataset_doc
        self._dist_loss_scale_factor = (
            dist.get_world_size() if self.args.negatives_x_device else 1
        )

    def _save(self, output_dir: Optional[str] = None):
        """Save model/trainer state and metrics."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]):
        """Prepare inputs for model."""
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self):
        """Get the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        """Get the evaluation dataloader."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
        """Compute loss."""
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def log(self, logs: Dict[str, float]):
        """Log metrics."""
        if "gen_r1_list" in self.model.__dict__:
            gen_r1_train = np.mean(self.model.gen_r1_list)
            logs["gen_r1_train"] = round(gen_r1_train, 4)
            self.model.gen_r1_list = []
        super().log(logs)
        return

    def training_step(self, *args):
        """Training step."""
        return (
            super(GLENP2Trainer, self).training_step(*args)
            / self._dist_loss_scale_factor
        )

    def _load_state_dict_in_model(self, state_dict: Dict[str, Any]):
        """Load state dict in model."""
        load_result = self.model.lm_p.load_state_dict(state_dict, strict=False)
        if len(load_result.missing_keys) != 0:
            if self.model.lm_p._keys_to_ignore_on_save is not None and set(
                load_result.missing_keys
            ) == set(self.model.lm_p._keys_to_ignore_on_save):
                self.model.lm_p.tie_weights()
            else:
                logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

        load_result = self.model.lm_q.load_state_dict(state_dict, strict=False)
        if len(load_result.missing_keys) != 0:
            if self.model.lm_q._keys_to_ignore_on_save is not None and set(
                load_result.missing_keys
            ) == set(self.model.lm_q._keys_to_ignore_on_save):
                self.model.lm_q.tie_weights()
            else:
                logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def make_doc_id(self):
        """Process document and make doc id."""
        all_ids = []
        dataloader = DataLoader(
            self.eval_dataset_doc,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        for _, batch in enumerate(
            tqdm(dataloader, dynamic_ncols=True, desc="document processing")
        ):
            oldids, texts, preds, out_logits = self.model.make_doc_id(batch)

            out_logits = np.round(out_logits.astype(np.float64), 4)
            for oldid, text, pred, out_logit in zip(
                oldids, texts, preds, out_logits.tolist()
            ):
                out_logit = "<->".join([str(x) for x in out_logit])
                all_ids.append([oldid, pred, out_logit, text])

        with open(self.model.model_args.docid_file_name, "w") as f:
            for oldid, pred, out_logit, text in all_ids:
                f.write(f"{oldid}\t{pred}\t{out_logit}\t{text}\n")

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Run evaluation and returns metrics.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Datastet to evaluate.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model that should be ignored when gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output["metrics"].update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output["num_samples"],
                num_steps=math.ceil(output["num_samples"] / total_batch_size),
            )
        )

        self.log(output["metrics"])
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output["metrics"]
        )
        self._memory_tracker.stop_and_update_metrics(output["metrics"])
        return output["metrics"]

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # Make id for GLEN evaluation
        self.make_doc_id()
        self.model.build_tree(
            log_step=self.state.global_step, log_file=args.eval_log_file
        )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        observed_num_examples = 0

        # Main evaluation loop
        metrics = {}
        texts, preds, labels, ranks = [], [], [], []
        for step, inputs in enumerate(
            tqdm(dataloader, dynamic_ncols=True, desc="inference")
        ):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            text, pred, label, rank = self.model.evaluation_step(inputs)
            pred = [",".join(p) for p in pred]

            texts += text
            preds += pred
            labels += label
            ranks += rank

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        # Save result
        res = pd.DataFrame(
            list(zip(texts, preds, labels, ranks)),
            columns=["query", "pred", "gt", "rank"],
        )
        res["rank"] = res["rank"].astype(int)
        res.sort_values(by=["query", "rank"], ascending=True, inplace=True)
        res1 = res.loc[res["rank"] == 1]
        res1.to_csv(args.res1_save_path, mode="w", sep="\t", header=None, index=False)

        # Evaluation
        metrics.update(compute_recall(args=args, cutoff=args.recall_num))
        metrics.update(compute_mrr(args=args, cutoff=args.mrr_num))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return {
            "num_samples": observed_num_examples,
            "metrics": metrics,
        }

    def log(self, logs: Dict[str, float]):
        super().log(logs)
        return


class GLENP2Trainer_GC(GLENP2Trainer):
    def __init__(self, *args, **kwargs):
        logger.info("Initializing Gradient Cache Trainer")
        if not _grad_cache_available:
            raise ValueError(
                "Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache."
            )
        super(GLENP2Trainer_GC, self).__init__(*args, **kwargs)

        loss_fn_cls = GLENLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def training_step(self, model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {"query": queries}, {"passage": passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed, model=model)

        return loss / self._dist_loss_scale_factor
