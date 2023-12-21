import json
import logging
import os
import sys
import time
import torch
import wandb
import warnings

import pandas as pd

from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig,
)

from tevatron.arguments import (
    GLENP2ModelArguments as ModelArguments,
    GLENP2DataArguments as DataArguments,
    GLENP2TrainingArguments as TrainingArguments,
)
from tevatron.datasets import GLENP2TrainDataset, GLENP2EncodeDataset, QPCollator
from tevatron.modeling import GLENP2Model
from tevatron.trainer import GLENP2Trainer, GLENP2Trainer_GC as GCTrainer

logger = logging.getLogger(__name__)
YOUR_API_KEY = ""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(action="ignore")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    assert model_args.model_name_or_path.startswith(
        "t5-"
    ), "Only T5- are supported for GLEN"

    if model_args.model_name_or_path == "t5-large":
        model_args.num_layers = 24
        model_args.num_decoder_layers = 24
        model_args.d_ff = 4096
        model_args.d_model = 1024
        model_args.num_heads = 16
        model_args.d_kv = 64

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
        num_layers=model_args.num_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        d_ff=model_args.d_ff,
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        decoder_start_token_id=0,  # 1,
        output_past=True,
        d_kv=model_args.d_kv,
        dropout_rate=model_args.dropout_rate,
        tie_word_embeddings=model_args.tie_word_embeddings,
    )
    model = GLENP2Model.build(
        model_args,
        training_args,
        tokenizer=tokenizer,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.grad_cache:
        assert data_args.negative_passage_type in [
            "self",
            "random",
            "hard",
        ], "grad_cache not implemented for other negative_passage_type"

    # Training dataset
    if data_args.dataset_name in ["nq320k", "marco_passage"]:
        train_dataset = GLENP2TrainDataset(data_args=data_args, tokenizer=tokenizer)
    else:
        raise NotImplementedError(
            f"dataset_name {data_args.dataset_name} not implemented"
        )

    # Evaluation
    if training_args.do_eval and data_args.dataset_name in ["nq320k", "marco_passage"]:
        assert (
            training_args.eval_accumulation_steps is None
        ), "eval_accumulation_steps not implemented"
        assert data_args.test100 == 0, "test100 not available for do_eval"
        model_args.max_output_length = model_args.num_multi_vectors + 1

        # dataset
        eval_dataset = GLENP2EncodeDataset(
            data_args=data_args,
            tokenizer=tokenizer,
            max_len=data_args.q_max_len,
            task="infer_qry",
        )
        eval_dataset_doc = GLENP2EncodeDataset(
            data_args=data_args,
            tokenizer=tokenizer,
            max_len=data_args.p_max_len,
            task="make_id",
        )

        # Set docid_file_name
        if model_args.docid_file_name == "":
            model_args.docid_file_name = f"{model.__class__.__name__}_len_{data_args.p_max_len}_{data_args.dataset_name}"
        model_args.docid_file_name = os.path.join(
            training_args.output_dir, model_args.docid_file_name + ".tsv"
        )

        # Set res1_save_path
        if training_args.res1_save_path == "":
            training_args.res1_save_path = f"{model.__class__.__name__}_len_{data_args.p_max_len}_{data_args.dataset_name}_res1"
        training_args.res1_save_path = os.path.join(
            training_args.output_dir, training_args.res1_save_path + ".tsv"
        )

        # Set evaluation log file path
        training_args.eval_log_file = os.path.join(
            training_args.output_dir, "eval_gen_full.txt"
        )

        # Load unseen query, seen query set
        if data_args.dataset_name == "nq320k":
            seen_query_df = pd.read_csv(
                "data/nq320k/GTQ_NQ_dev_seen.tsv", sep="\t", dtype=str
            )
            unseen_query_df = pd.read_csv(
                "data/nq320k/GTQ_NQ_dev_unseen.tsv", sep="\t", dtype=str
            )
            training_args.unseen_query_set = set(unseen_query_df["query"])
            training_args.seen_query_set = set(seen_query_df["query"])
            print(
                f"> Loading unseen query (#:{len(training_args.unseen_query_set)}) and seen query (#:{len(training_args.seen_query_set)})"
            )

        # Set metric cutoff
        training_args.recall_num = [1, 10, 100]
        training_args.ndcg_num = [1, 10, 100]
        training_args.mrr_num = [10, 100]

        # remain only if smaller than model_args.num_return_sequences
        training_args.recall_num = [
            x for x in training_args.recall_num if x <= model_args.num_return_sequences
        ]
        training_args.ndcg_num = [
            x for x in training_args.ndcg_num if x <= model_args.num_return_sequences
        ]
        training_args.mrr_num = [
            x for x in training_args.mrr_num if x <= model_args.num_return_sequences
        ]

    else:
        eval_dataset, eval_dataset_doc = None, None

    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer_cls = GCTrainer if training_args.grad_cache else GLENP2Trainer

    # Initialize trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_dataset_doc=eval_dataset_doc,
        data_collator=QPCollator(
            tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len
        ),
    )
    trainer.data_args = data_args
    train_dataset.trainer = trainer
    model.trainer = trainer

    # Set masking for special tokens in evaluation decoding
    model.tokenizer = tokenizer
    if model_args.mask_special_tokens_for_decoding:
        special_token_ids = tokenizer.all_special_ids
        model_args.special_token_ids = [
            x
            for x in special_token_ids
            if x
            not in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ]
        ]

    # Load SentenceT5Model checkpoint
    if model_args.load_pretrained_st5_checkpoint is not None:
        print(
            f"> Restoring parameters from checkpoint {model_args.load_pretrained_st5_checkpoint}"
        )

        if model_args.load_pretrained_st5_checkpoint.endswith(
            ".ckpt"
        ) or model_args.load_pretrained_st5_checkpoint.endswith(".bin"):
            state_dict = torch.load(model_args.load_pretrained_st5_checkpoint)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = torch.load(
                os.path.join(
                    model_args.load_pretrained_st5_checkpoint, "pytorch_model.bin"
                )
            )

        if model_args.untie_encoder:
            model.lm_q.load_state_dict(state_dict, strict=False)
            model.lm_p.load_state_dict(state_dict, strict=False)
        else:
            model.lm_q.load_state_dict(state_dict, strict=False)
            model.lm_p = model.lm_q
    print(
        f"> Restored parameters from checkpoint {model_args.load_pretrained_st5_checkpoint}"
    )

    # Weight tying
    if "lm_head.weight" in model.lm_p.state_dict() and model_args.untie_encoder:
        state_dict = model.lm_p.state_dict()
        model.lm_p.shared.weight.data.copy_(state_dict["shared.weight"])
        model.lm_p.lm_head.weight.data.copy_(model.lm_p.shared.weight.data)
        state_dict = model.lm_q.state_dict()
        model.lm_q.shared.weight.data.copy_(state_dict["shared.weight"])
        model.lm_q.lm_head.weight.data.copy_(model.lm_q.shared.weight.data)
    elif "lm_head.weight" in model.lm_p.state_dict() and not model_args.untie_encoder:
        state_dict = model.lm_p.state_dict()
        model.lm_p.shared.weight.data.copy_(state_dict["shared.weight"])
        model.lm_p.lm_head.weight.data.copy_(model.lm_p.shared.weight.data)
        model.lm_q = model.lm_p

    # Save args and tokenizer
    if trainer.is_world_process_zero():
        # Save args and tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, "model_args.json"), "w") as f:
            json.dump(model_args.__dict__, f, indent=4)
        with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as f:
            json.dump(data_args.__dict__, f, indent=4)

        # Report to wandb
        if YOUR_API_KEY != "":
            training_args.report_to = "wandb"
            os.environ["WANDB_API_KEY"] = YOUR_API_KEY

            important_info_list = [str(data_args.dataset_name.replace("/", "_"))]
            if data_args.dataset_name in ["nq320k", "marco_passage"]:
                important_info_list += [str(data_args.query_type)]
            important_info_list += [str(model.__class__.__name__)]
            important_info_str = "_".join(important_info_list)

            wandb_tag = (
                training_args.wandb_tag.split(",") if training_args.wandb_tag else []
            )
            wandb_name = f'{time.strftime("%Y%m%d-%H%M%S")}-{important_info_str}'
            wandb.init(
                project=training_args.project_name,
                name=wandb_name,
                settings=wandb.Settings(save_code=True, code_dir="."),
                tags=wandb_tag,
            )

    # Train
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    main()
