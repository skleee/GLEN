import os
import math
import time
import logging
import pandas as pd

import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union

from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.trainer_pt_utils import find_batch_size
from transformers.file_utils import is_apex_available

from tevatron.metrics import compute_recall, compute_mrr

logger = logging.getLogger(__name__)

if is_apex_available():
    from apex import amp


class GLENP1Trainer(Trainer):
    def __init__(self, eval_dataset_doc: Optional[Dataset] = None, *args, **kwargs):
        super(GLENP1Trainer, self).__init__(*args, **kwargs)
        self.model_args = self.model.model_args
        self.eval_dataset_doc = eval_dataset_doc
        self._dist_loss_scale_factor = (
            dist.get_world_size() if self.args.negatives_x_device else 1
        )
        train_dataset = self.train_dataset
        self.t_total = (
            (
                len(train_dataset)
                // (self.args.train_batch_size * max(1, self.args.n_gpu))
            )
            // self.args.gradient_accumulation_steps
            * float(self.args.num_train_epochs)
        )
        self.args.eval_steps = max(
            int(
                (self.t_total // self.args.num_train_epochs)
                * self.args.val_check_interval
                / self.args.n_gpu
            ),
            1,
        )
        self.args.save_steps = self.args.eval_steps

        print(
            f"t_total: {self.t_total}, eval_steps: {self.args.eval_steps}, save_steps: {self.args.save_steps}"
        )

    def _save(self, output_dir: Optional[str] = None):
        """Save model/training states to disk"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]):
        """Prepare inputs before feeding them to the model."""
        prepared = {}
        for key, x in inputs.items():
            if isinstance(x, torch.Tensor):
                prepared[key] = x.to(self.args.device)
            else:
                prepared[key] = super()._prepare_inputs(x)
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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        Set learning rate for encoder and decoder separately.
        """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay))
                    and (n.startswith(("hf_model.shared.", "hf_model.encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay))
                    and (not n.startswith(("hf_model.shared.", "hf_model.encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                    and (n.startswith(("hf_model.shared.", "hf_model.encoder.")))
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                    and (not n.startswith(("hf_model.shared.", "hf_model.encoder.")))
                ],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.t_total,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        encoder_outputs=None,
        input_mask=None,
    ):
        """Compute the loss of the model on a batch of inputs."""
        lm_labels, target_mask = inputs["target_ids"], inputs["target_mask"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = model.forward(
            input_ids=inputs["source_ids"],
            aug_input_ids=inputs["aug_source_ids"],
            attention_mask=inputs["source_mask"],
            aug_attention_mask=inputs["aug_source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=target_mask,
            encoder_outputs=encoder_outputs,
            input_mask=input_mask,
        )
        loss = outputs.loss
        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ):
        """
        Perform a training step on a batch of inputs.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def _load_state_dict_in_model(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load the state dict in the model

        Args:
            state_dict (`Dict[str, torch.Tensor]`):
                The state dict to load in the model.

        Returns:
            None
        """
        load_result = self.model.hf_model.load_state_dict(state_dict, strict=False)
        if len(load_result.missing_keys) != 0:
            if self.model.hf_model._keys_to_ignore_on_save is not None and set(
                load_result.missing_keys
            ) == set(self.model.hf_model._keys_to_ignore_on_save):
                self.model.hf_model.tie_weights()
            else:
                logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def make_doc_id(self):
        """Process all documents and make doc id file"""
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
        """Evaluation loop, shared by `evaluate()` and `predict()`."""

        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
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
        for step, inputs in enumerate(dataloader):
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
        if metric_key_prefix != "":
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
