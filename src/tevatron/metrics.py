import os
import pandas as pd

from typing import List, Optional
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def compute_recall(args, cutoff: List[int] = [1, 10, 100]):
    """
    Compute recall@k

    Args:
        args: arguments
        cutoff: list of cutoffs
    Returns:
        metrics: dict of metrics
    """
    q_gt, q_pred = {}, {}
    with open(args.res1_save_path, "r") as f:
        prev_q = ""
        for line in f.readlines():
            query, pred, gt, rank = line[:-1].split("\t")
            if query != prev_q:
                q_pred[query] = pred.split(",")
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = gt.split(",")
                q_gt[query] = set(q_gt[query])

    do_seen_unseen = (
        True
        if args.unseen_query_set is not None and args.seen_query_set is not None
        else False
    )
    metrics = {}

    lines = []
    lines.append("#####################")
    for i in cutoff:
        recall_k, seen_recall_k, unseen_recall_k = [], [], []
        for q in q_pred:
            tmp_recall = (
                len(set(q_gt[q]) & set(q_pred[q][: int(i)])) / len(q_gt[q])
                if len(q_gt[q]) > 0
                else 0
            )
            recall_k.append(tmp_recall)
            if do_seen_unseen:
                if q in args.seen_query_set:
                    tmp_recall = (
                        len(set(q_gt[q]) & set(q_pred[q][: int(i)])) / len(q_gt[q])
                        if len(q_gt[q]) > 0
                        else 0
                    )
                    seen_recall_k.append(tmp_recall)
                elif q in args.unseen_query_set:
                    tmp_recall = (
                        len(set(q_gt[q]) & set(q_pred[q][: int(i)])) / len(q_gt[q])
                        if len(q_gt[q]) > 0
                        else 0
                    )
                    unseen_recall_k.append(tmp_recall)

        recall_avg = sum(recall_k) / len(recall_k)
        if do_seen_unseen:
            seen_recall_avg = (
                sum(seen_recall_k) / len(seen_recall_k) if len(seen_recall_k) > 0 else 0
            )
            unseen_recall_avg = (
                sum(unseen_recall_k) / len(unseen_recall_k)
                if len(unseen_recall_k) > 0
                else 0
            )
            metrics.update(
                {
                    f"recall@{i}": recall_avg,
                    f"recall_unseen@{i}": unseen_recall_avg,
                    f"recall_seen@{i}": seen_recall_avg,
                }
            )
            lines.append(
                f"recall@{i} : {recall_avg:.4f} | recall_unseen@{i} : {unseen_recall_avg:.4f} | recall_seen@{i} : {seen_recall_avg:.4f}"
            )
        else:
            metrics.update({f"recall@{i}": recall_avg})
            lines.append(f"recall@{i} : {recall_avg:.4f}")
    lines.append("-------------------------")
    print("\n".join(lines))

    return metrics


def compute_mrr(args, cutoff: List[int] = [10, 100]):
    """
    Compute MRR@k

    Args:
        args: arguments
        cutoff: list of cutoffs
    Returns:
        metrics: dict of metrics
    """
    q_gt, q_pred = {}, {}
    with open(args.res1_save_path, "r") as f:
        prev_q = ""
        for line in f.readlines():
            query, pred, gt, rank = line[:-1].split("\t")
            if query != prev_q:
                q_pred[query] = pred.split(",")
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = gt.split(",")
                q_gt[query] = set(q_gt[query])

    do_seen_unseen = (
        True
        if args.unseen_query_set is not None and args.seen_query_set is not None
        else False
    )
    metrics = {}
    lines = []
    for i in cutoff:
        mrr_k, seen_mrr_k, unseen_mrr_k = [], [], []
        for query in q_pred:
            score = 0
            for j, p in enumerate(q_pred[query][: int(i)]):
                if p in q_gt[query]:
                    score = 1 / (j + 1)
                    break
            mrr_k.append(score)

            if do_seen_unseen:
                if query in args.seen_query_set:
                    seen_mrr_k.append(score)
                elif query in args.unseen_query_set:
                    unseen_mrr_k.append(score)

        mrr = sum(mrr_k) / len(mrr_k)
        if do_seen_unseen:
            seen_mrr = sum(seen_mrr_k) / len(seen_mrr_k) if len(seen_mrr_k) > 0 else 0
            unseen_mrr = (
                sum(unseen_mrr_k) / len(unseen_mrr_k) if len(unseen_mrr_k) > 0 else 0
            )
            metrics.update(
                {
                    f"MRR@{i}": mrr,
                    f"MRR_unseen@{i}": unseen_mrr,
                    f"MRR_seen@{i}": seen_mrr,
                }
            )
            lines.append(
                f"MRR@{i} : {mrr:.4f} | MRR_unseen@{i} : {unseen_mrr:.4f} | MRR_seen@{i} : {seen_mrr:.4f}"
            )
        else:
            metrics.update({f"MRR@{i}": mrr})
            lines.append(f"MRR@{i} : {mrr:.4f}")

    print("\n".join(lines))
    return metrics


def evaluate_beir(args, tokenizer: AutoTokenizer, dataset: Optional[Dataset]):
    """
    Evaluate BEIR dataset using beir library

    Args:
        args: arguments
        tokenizer: tokenizer
        dataset: dataset
    Returns:
        metrics: dict of metrics
    """
    q_gt, q_pred = {}, {}

    with open(args.res1_save_path, "r") as f:
        prev_q = ""
        for line in f.readlines():
            query, pred, gt, rank = line[:-1].split("\t")
            if query != prev_q:
                q_pred[query] = pred.split(",")
                q_pred[query] = q_pred[query]
                prev_q = query
            if query in q_gt:
                if len(q_gt[query]) <= 100:
                    q_gt[query].add(gt)
            else:
                q_gt[query] = gt.split(",")
                q_gt[query] = set(q_gt[query])

    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval

    data_path = os.path.join("data/BEIR_dataset", args.dataset_name)
    _, _, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    fname = os.path.join(data_path, "dev_doc_newid.tsv")
    df = pd.read_csv(fname, encoding="utf-8", sep="\t", dtype=str).loc[
        :, ["query", "queryid"]
    ]

    df_unique_q = df.drop_duplicates(subset=["query", "queryid"])

    query2qid = {}
    for query, qid in df_unique_q[["query", "queryid"]].values:
        input_ = dataset.clean_text(query)
        output_ = tokenizer.batch_encode_plus(
            [input_],
            max_length=156,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        query = tokenizer.decode(
            output_["input_ids"][0].numpy(), skip_special_tokens=True
        )
        query2qid[query] = qid

    retriever = EvaluateRetrieval(None, score_function="dot")

    results = defaultdict(dict)
    for q in q_pred:
        qid = query2qid[q]
        for rank, d in enumerate(q_pred[q]):
            score = 1 / (rank + 1)
            oldid = d.split("<->")[0]
            results[qid][oldid] = score

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, args.ndcg_num)

    metrics = {}
    print("#####################")
    for k in args.ndcg_num:
        metrics.update({f"NDCG@{k}": ndcg[f"NDCG@{k}"]})
        score = ndcg[f"NDCG@{k}"]
        print(f"NDCG@{k} : {score}")
    print("#####################")
    for k in args.recall_num:
        metrics.update({f"Recall@{k}": recall[f"Recall@{k}"]})
        score = recall[f"Recall@{k}"]
        print(f"Recall@{k} : {score}")
    print("#####################")

    return metrics
