import json
import logging
import os
import sys
import time
import torch
import warnings

import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)

from tevatron.arguments import (
    GLENP1ModelArguments as ModelArguments,
    GLENP1DataArguments as DataArguments,
    GLENP1TrainingArguments as TrainingArguments,
)
from tevatron.datasets import GLENP1EncodeDataset
from tevatron.modeling import GLENP1Model, T5Config
from tevatron.metrics import compute_recall, compute_mrr, evaluate_beir
from tevatron.tree import TreeBuilder

logger = logging.getLogger(__name__)
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

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU is not supported.")

    if os.path.exists(os.path.join(model_args.infer_dir, "model_args.json")):
        print(
            f"> Load model arguments from {os.path.join(model_args.infer_dir, 'model_args.json')}"
        )
        with open(os.path.join(model_args.infer_dir, "model_args.json"), "r") as f:
            model_args_dict = json.load(f)
        model_args = ModelArguments(**model_args_dict)
    else:
        print(f"> Not found model arguments from {os.path.join(model_args.infer_dir)}")

    # Assert if num_return_sequences is larger than 1
    assert (
        model_args.num_return_sequences > 1
    ), "num_return_sequences should be larger than 1"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    set_seed(training_args.seed)

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
    decode_vocab_size = 32128 if len(tokenizer) == 32100 else len(tokenizer)
    config = T5Config(
        num_layers=model_args.num_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        d_ff=model_args.d_ff,
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        decoder_start_token_id=0,
        output_past=True,
        d_kv=model_args.d_kv,
        dropout_rate=model_args.dropout_rate,
        decode_vocab_size=decode_vocab_size,
        tie_word_embeddings=model_args.tie_word_embeddings,
        tie_decode_embeddings=model_args.tie_decode_embeddings,
        Rdrop=model_args.Rdrop,
        input_dropout=model_args.input_dropout,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
    model = GLENP1Model.load(
        model_args=model_args,
        tokenizer=tokenizer,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Set result file name
    if not os.path.exists(model_args.logs_dir):
        os.makedirs(model_args.logs_dir)

    infer_ckpt_info = "_".join(model_args.infer_dir.split("/"))
    training_args.res1_save_path = os.path.join(
        model_args.logs_dir,
        f'{time.strftime("%Y%m%d-%H%M%S")}_res1_recall{model_args.num_return_sequences}_{data_args.dataset_name}_{str(model.__class__.__name__)}_{infer_ckpt_info}.tsv',
    )

    # load checkpoint
    if model_args.infer_ckpt:
        ckpt_path = model_args.infer_ckpt
    else:
        ckpt_path = os.path.join(model_args.infer_dir, "pytorch_model.bin")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.hf_model.load_state_dict(state_dict, strict=False)

    if "lm_head.weight" in model.hf_model.state_dict():
        if "shared.weight" in state_dict:
            model.hf_model.shared.weight.data.copy_(state_dict["shared.weight"])
        elif "model.shared.weight" in state_dict:
            model.hf_model.shared.weight.data.copy_(state_dict["model.shared.weight"])
        else:
            raise Exception("shared.weight not found in state_dict")
        model.hf_model.lm_head.weight.data.copy_(model.hf_model.shared.weight.data)

    del state_dict

    # Load dataset
    if data_args.dataset_name in ["nq320k", "marco_passage", "nfcorpus", "arguana"]:
        encode_dataset = GLENP1EncodeDataset(
            data_args=data_args,
            tokenizer=tokenizer,
            max_len=data_args.max_input_length,
            task="infer_qry",
        )
    else:
        raise NotImplementedError(f"{data_args.dataset_name} is not supported")

    max_output_length = model_args.max_output_length

    # Mask special tokens for decoding
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

    # Build tree
    docid_file_name = (
        "/".join(model_args.infer_dir.split("/")[:-1])
        + "/"
        + model_args.docid_file_name
        + ".tsv"
    )
    docid_df = pd.read_csv(
        docid_file_name,
        sep="\t",
        names=["oldid", "docid", "docid_logit", "text"],
        dtype={"oldid": str, "docid": str, "docid_logit": str, "text": str},
    ).loc[:, ["oldid", "docid", "docid_logit"]]

    num_uniques = len(set(docid_df.docid))
    unique_ratio = num_uniques / len(docid_df)
    num_unique_tokens = [0] * 10
    docid2num_docs = dict(docid_df.docid.value_counts())
    for docid in docid_df.docid.unique():
        tokens = docid.split("<->")
        num_unique_token = len(set(tokens))
        num_unique_tokens[num_unique_token - 1] += 1
    print(f"num_uniques: {num_uniques}/{len(docid_df)} ({unique_ratio*100:.2f}%)")
    print("[Frequent Collision]", docid_df.docid.value_counts()[:5], sep="\n")
    print(f"distribution of number of unique tokens: {num_unique_tokens}")

    docid2oldids = defaultdict(list)
    oldid2docid_logit = dict()
    for docid, oldid, docid_logit in docid_df[["docid", "oldid", "docid_logit"]].values:
        docid2oldids[docid].append(oldid)
        oldid2docid_logit[oldid] = torch.tensor(
            [float(x) for x in docid_logit.split("<->")]
        )

    oldid2docid = dict(zip(docid_df.oldid, docid_df.docid))
    if model_args.tree == 1:
        builder = TreeBuilder()
        all_id = []
        for docid in list(oldid2docid.values()):
            toks = docid.split("<->")
            toks = tokenizer.convert_tokens_to_ids(toks)
            if len(toks) != max_output_length - 1:
                print(toks, docid, "is not equal to max_output_length")
                toks = toks[: max_output_length - 1]
            all_id.append(toks)
            builder.add(toks)
        model.root = builder.build()

    # Store tree in model
    model.docid2num_docs = docid2num_docs
    model.docid2oldids = docid2oldids
    model.oldid2docid_logit = oldid2docid_logit
    model.oldid2docid = oldid2docid

    # Infer query
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = model.to(training_args.device)
    model.eval()

    texts, preds, labels, ranks = [], [], [], []
    for batch in tqdm(encode_loader, dynamic_ncols=True, desc="query processing"):
        text, pred, label, rank = model.evaluation_step(batch, predefined_id=False)
        pred = [",".join(p) for p in pred]

        texts += text
        preds += pred
        labels += label
        ranks += rank

    # Save result
    res = pd.DataFrame(
        list(zip(texts, preds, labels, ranks)), columns=["query", "pred", "gt", "rank"]
    )
    res["rank"] = res["rank"].astype(int)
    res.sort_values(by=["query", "rank"], ascending=True, inplace=True)
    res1 = res.loc[res["rank"] == 1]
    res1.to_csv(
        training_args.res1_save_path, mode="w", sep="\t", header=None, index=False
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

    # set training_args attributes for compute_recall, compute_mrr
    training_args.dataset_name = data_args.dataset_name
    training_args.unseen_query_set, training_args.seen_query_set = None, None

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

        compute_recall(training_args, cutoff=training_args.recall_num)
        compute_mrr(training_args, cutoff=training_args.mrr_num)
    elif data_args.dataset_name == "marco_passage":
        compute_recall(training_args, cutoff=training_args.recall_num)
        compute_mrr(training_args, cutoff=training_args.mrr_num)
    else:
        evaluate_beir(training_args, tokenizer, encode_dataset)


if __name__ == "__main__":
    main()
