import os
import torch

import numpy as np
import pandas as pd

from collections import defaultdict
from time import time
from tqdm import tqdm

from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset

from tevatron.trainer import GLENP1Trainer
from tevatron.arguments import GLENP1DataArguments

import logging

logger = logging.getLogger(__name__)


class GLENP1TrainDataset(Dataset):
    def __init__(
        self,
        data_args: GLENP1DataArguments,
        tokenizer: PreTrainedTokenizer,
        trainer: GLENP1Trainer = None,
    ):
        self.data_args = data_args

        assert self.data_args.dataset_name in ["nq320k", "marco_passage"]

        self.dataset, self.docid2doc, self.docid2oldid = load_data(
            data_args, return_docid2doc=True
        )
        self.oldid2docid = {v: k for k, v in self.docid2oldid.items()}
        self.docid_list = list(self.docid2oldid.keys())

        self.trainer = trainer
        self.tokenizer = tokenizer
        self.input_length = self.data_args.max_input_length
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        ] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

        if self.data_args.test100:
            self.dataset = self.dataset[:100]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def clean_text(text):
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")
        return text

    def convert_to_features(self, example_batch, length_constraint):
        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=length_constraint,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return output_

    def __getitem__(self, index):
        inputs = self.dataset[index]

        query, target, rank, neg_target, aug_query, oldid = (
            inputs.get("query"),
            inputs.get("docid"),
            inputs.get("rank"),
            inputs.get("neg_docid_list"),
            inputs.get("aug_query_list"),
            inputs.get("oldid", -1),
        )

        if self.data_args.aug_query and len(aug_query) >= 1:
            aug_query = np.random.choice(aug_query, 1)[0]
        else:
            aug_query = ""

        source = self.convert_to_features(query, self.input_length)
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        aug_source = self.convert_to_features(aug_query, self.input_length)
        aug_source_ids = aug_source["input_ids"].squeeze()
        aug_source_mask = aug_source["attention_mask"].squeeze()

        lm_labels = torch.zeros(self.data_args.max_output_length, dtype=torch.long)

        target_id_int = self.tokenizer.convert_tokens_to_ids(target.split("-"))

        lm_labels[: len(target_id_int)] = torch.LongTensor(target_id_int)
        lm_labels[len(target_id_int)] = 1

        decoder_attention_mask = lm_labels.clone()
        decoder_attention_mask[decoder_attention_mask != 0] = 1

        target_ids = lm_labels
        target_mask = decoder_attention_mask

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "aug_source_ids": aug_source_ids,
            "aug_source_mask": aug_source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "rank": rank,
            "oldid": oldid,
        }


class GLENP1EncodeDataset(Dataset):
    def __init__(
        self,
        data_args: GLENP1DataArguments,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 156,
        task: str = "make_id",
    ):
        assert task in ["make_id", "infer_qry"]
        assert data_args.dataset_name in [
            "nq320k",
            "marco_passage",
            "nfcorpus",
            "arguana",
        ]

        self.data_args = data_args

        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.vocabs = set(self.tokenizer.get_vocab().keys())
        for token in [
            self.tokenizer.eos_token,
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
            self.tokenizer.mask_token,
        ] + tokenizer.additional_special_tokens:
            if token is not None:
                self.vocabs.remove(token)

        self.dataset = self.load_data_infer()

        if self.data_args.test100:
            self.dataset = self.dataset[:100]

        if self.task == "make_id":
            self.load_oldid2t5tokenizeddoc()

    @staticmethod
    def clean_text(text):
        text = text.replace("\n", "")
        text = text.replace("``", "")
        text = text.replace('"', "")
        return text

    def convert_to_features(self, example_batch):
        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.encode_plus(
            input_,
            max_length=self.max_len,
            truncation="only_first",
            padding=False,
            return_token_type_ids=False,
        )
        return output_

    def convert_to_features_batch_pt(self, example_batch):
        input_ = self.clean_text(example_batch)
        output_ = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return output_

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.task == "make_id":
            inputs = self.dataset[index]
            oldid, doc_content = inputs["oldid"], inputs["doc_content"]
            if self.oldid2t5tokenizeddoc.get(oldid) is None:
                source = self.convert_to_features_batch_pt(doc_content)
            else:
                source = self.oldid2t5tokenizeddoc[oldid]
                source["input_ids"] = source["input_ids"][:, : self.max_len]
                source["attention_mask"] = source["attention_mask"][:, : self.max_len]
            source_ids = source["input_ids"].squeeze()
            src_mask = source["attention_mask"].squeeze()
            return {"source_ids": source_ids, "source_mask": src_mask, "oldids": oldid}

        else:
            inputs = self.dataset[index]
            query, oldid, target, rank, neg_target, aug_query = (
                inputs.get("query"),
                inputs.get("oldid", -1),
                inputs.get("target"),
                inputs.get("rank"),
                inputs.get("neg_target"),
                inputs.get("aug_query"),
            )
            if self.data_args.aug_query and len(aug_query) >= 1:
                aug_query = np.random.choice(aug_query, 1)[0]
            else:
                aug_query = ""

            aug_source = self.convert_to_features_batch_pt(aug_query)
            aug_source_ids = aug_source["input_ids"].squeeze()
            aug_source_mask = aug_source["attention_mask"].squeeze()

            text = query

            source = self.convert_to_features_batch_pt(text)
            source_ids = source["input_ids"].squeeze()
            source_mask = source["attention_mask"].squeeze()

            lm_labels = torch.zeros(self.max_len, dtype=torch.long)
            target_ids = lm_labels
            target_mask = lm_labels

            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "aug_source_ids": aug_source_ids,
                "aug_source_mask": aug_source_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "rank": rank,
                "oldid": oldid,
            }

    def load_data_infer(self):
        args = self.data_args

        # load oldid2newid file
        if args.dataset_name == "nq320k":
            newid_df = pd.read_csv(
                f"data/nq320k/ID_NQ_{args.id_class}.tsv", sep="\t", dtype=str
            )
            if args.small_set in [1, 10]:
                small_id_df = pd.read_csv(
                    f"data/nq320k/SMALL_NQ_{args.small_set}_oldid.tsv",
                    sep="\t",
                    dtype=str,
                )
                newid_df = newid_df[newid_df["oldid"].isin(small_id_df["oldid"])]
        elif args.dataset_name == "marco_passage":
            newid_df = pd.read_csv(
                f"data/marco_passage/ID_MARCO_{args.id_class}.tsv",
                sep="\t",
                dtype=str,
            )
            if args.small_set in [1]:
                small_id_df = pd.read_csv(
                    f"data/marco_passage/SMALL_MARCO_{args.small_set}_oldid.tsv",
                    sep="\t",
                    dtype=str,
                )
                newid_df = newid_df[newid_df["oldid"].isin(small_id_df["oldid"])]
        elif args.dataset_name in ["nfcorpus", "arguana"]:
            newid_df = pd.read_csv(
                f"data/BEIR_dataset/ID_{args.dataset_name}_{args.id_class}.tsv",
                sep="\t",
                dtype=str,
            )
        oldid2newid = dict(zip(newid_df["oldid"], newid_df[args.id_class]))

        if self.task == "infer_qry":
            if args.dataset_name == "nq320k":
                fname = f"data/nq320k/GTQ_NQ_dev.tsv"
            elif args.dataset_name == "marco_passage":
                if args.small_set == 1:
                    fname = "data/marco_passage/GTQ_MARCO_valid.tsv"
                else:
                    fname = "data/marco_passage/GTQ_MARCO_dev.tsv"
            elif args.dataset_name in ["nfcorpus", "arguana"]:
                fname = f"data/BEIR_dataset/GTQ_{args.dataset_name}_dev.tsv"
            df = pd.read_csv(fname, encoding="utf-8", sep="\t", dtype=str)
            df.insert(1, args.id_class, df["oldid"].map(oldid2newid))

            if args.small_set in [1, 10]:
                df.dropna(axis=0, inplace=True)

            if args.dataset_name in ["nfcorpus", "arguana"]:
                # drop nan
                original_len = len(df)
                df.dropna(axis=0, inplace=True)
                print(f"drop nan: {original_len - len(df)}")

                # drop duplicate query
                original_len = len(df)
                df.drop_duplicates(subset=["query"], inplace=True)
                print(
                    f"drop duplicate query: {original_len - len(df)} (but all test samples will be evaluated)"
                )

            assert not df.isnull().values.any()

            result = []
            neg_docid_list, aug_query_list = [], []  # do not need for inference
            for _, row in df.iterrows():
                query = row["query"]
                rank1 = "".join([c for c in str(row[args.id_class])])
                docid = "-".join([c for c in str(row[args.id_class]).split("-")])
                list_sum = [(docid, 1)]
                if args.aug_query:
                    for i in range(20):
                        aug_query_list.append(augment(query))
                result.append(
                    {
                        "query": query,
                        "oldid": row["oldid"],
                        "target": rank1,
                        "rank": list_sum,
                        "neg_target": neg_docid_list,
                        "aug_query": aug_query_list,
                    }
                )
            print(
                f">> [Infer - query] Loaded {len(result)} examples from {args.dataset_name} dataset."
            )
            return tuple(result)
        else:
            if args.dataset_name == "nq320k":
                doc_filename = f"data/nq320k/DOC_NQ_title+content.tsv"
                df = pd.read_csv(doc_filename, encoding="utf-8", sep="\t", dtype=str)
                if "query" in df.columns:
                    df.rename(columns={"query": "doc_content"}, inplace=True)
                assert not df.isnull().values.any()
                result = []
                for _, row in df.iterrows():
                    result.append(
                        {"oldid": row["oldid"], "doc_content": row["doc_content"]}
                    )
            elif args.dataset_name == "marco_passage":
                print(f"> Reading MARCO passage dataset...")
                if args.small_set == 1:  # for validation
                    doc_filename = (
                        "data/marco_passage/DOC_MARCO_collection_valid.tsv"
                    )
                else:
                    doc_filename = (
                        "data/marco_passage/DOC_MARCO_collection.tsv"
                    )
                df = pd.read_csv(doc_filename, encoding="utf-8", sep="\t", dtype=str)
                result = []
                for _, row in df.iterrows():
                    result.append({"oldid": row["oldid"], "doc_content": row["query"]})
            elif args.dataset_name in ["nfcorpus", "arguana"]:
                print(f"> Reading BEIR-{args.dataset_name} dataset...")
                doc_filename = f"data/BEIR_dataset/DOC_{args.dataset_name}_corpus.tsv"
                df = pd.read_csv(doc_filename, encoding="utf-8", sep="\t", dtype=str)
                result = []
                for _, row in df.iterrows():
                    result.append(
                        {"oldid": row["oldid"], "doc_content": row["document"]}
                    )
            print(
                f">> [Infer - doc] Loaded {len(result)} examples from {args.dataset_name} dataset."
            )
            return tuple(result)

    def load_oldid2t5tokenizeddoc(self):
        self.oldid2t5tokenizeddoc = dict()
        if self.data_args.dataset_name == "nq320k":
            filename = f"data/nq320k/oldid2t5tokenizeddoc_{self.max_len}.npy"

            if os.path.exists(filename) and not self.data_args.test100:
                print(f"Loading {filename} ... ")
                st = time()
                all_tokenized_np = np.load(filename)
                oldid_np = all_tokenized_np[:, 0]
                input_ids_np = all_tokenized_np[:, 1 : self.max_len + 1]
                attention_mask_np = all_tokenized_np[:, self.max_len + 1 :]
                for i in range(len(oldid_np)):
                    self.oldid2t5tokenizeddoc[str(oldid_np[i])] = {
                        "input_ids": torch.tensor(
                            input_ids_np[i : i + 1], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            attention_mask_np[i : i + 1], dtype=torch.long
                        ),
                    }
                print(f"Load {filename} in {time() - st:.2f} seconds")

            elif not os.path.exists(filename) and not self.data_args.test100:
                print(f"Building {filename} ... ")
                for each_data in tqdm(
                    self.dataset, desc=f"Building {filename}", dynamic_ncols=True
                ):
                    oldid = each_data["oldid"]
                    doc_content = each_data["doc_content"]
                    input_ = self.clean_text(doc_content)
                    output_ = self.tokenizer.batch_encode_plus(
                        [input_],
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    self.oldid2t5tokenizeddoc[oldid] = output_

                oldid_np = np.array(list(self.oldid2t5tokenizeddoc.keys()))
                input_ids_np, attention_mask_np = [], []
                for oldid in oldid_np:
                    input_ids_np.append(
                        self.oldid2t5tokenizeddoc[oldid]["input_ids"].numpy()
                    )
                    attention_mask_np.append(
                        self.oldid2t5tokenizeddoc[oldid]["attention_mask"].numpy()
                    )
                input_ids_np = np.concatenate(input_ids_np, axis=0)
                attention_mask_np = np.concatenate(attention_mask_np, axis=0)

                all_tokenized_np = np.concatenate(
                    [
                        oldid_np.astype(int).reshape(-1, 1),
                        input_ids_np,
                        attention_mask_np,
                    ],
                    axis=1,
                )  # (#data, 1 + length_constraint + length_constraint)
                np.save(filename, all_tokenized_np)

        elif self.data_args.dataset_name == "marco_passage":
            filename = f"data/marco_passage/oldid2t5tokenizeddoc_{self.max_len}.npy"
            if self.data_args.small_set in [1]:
                filename = f"data/marco_passage/SMALL_MARCO_{self.data_args.small_set}_oldid2t5tokenizeddoc_{self.max_len}.npy"

            if os.path.exists(filename) and not self.data_args.test100:
                print(f"Loading {filename} ... ")
                st = time()
                all_tokenized_np = np.load(filename)
                oldid_np = all_tokenized_np[:, 0]
                input_ids_np = all_tokenized_np[:, 1 : self.max_len + 1]
                attention_mask_np = all_tokenized_np[:, self.max_len + 1 :]
                for i in range(len(oldid_np)):
                    self.oldid2t5tokenizeddoc[str(oldid_np[i])] = {
                        "input_ids": torch.tensor(
                            input_ids_np[i : i + 1], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            attention_mask_np[i : i + 1], dtype=torch.long
                        ),
                    }
                print(f"Loading {filename} in {time() - st:.2f} seconds")

            elif not os.path.exists(filename) and not self.data_args.test100:
                print(f"Building {filename} ... ")
                for each_data in tqdm(
                    self.dataset, desc=f"Building {filename}", dynamic_ncols=True
                ):
                    oldid = each_data["oldid"]
                    doc_content = each_data["doc_content"]

                    input_ = self.clean_text(doc_content)
                    output_ = self.tokenizer.batch_encode_plus(
                        [input_],
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    self.oldid2t5tokenizeddoc[oldid] = output_

                oldid_np = np.array(list(self.oldid2t5tokenizeddoc.keys()))
                input_ids_np, attention_mask_np = [], []
                for oldid in oldid_np:
                    input_ids_np.append(
                        self.oldid2t5tokenizeddoc[oldid]["input_ids"].numpy()
                    )
                    attention_mask_np.append(
                        self.oldid2t5tokenizeddoc[oldid]["attention_mask"].numpy()
                    )
                input_ids_np = np.concatenate(input_ids_np, axis=0)
                attention_mask_np = np.concatenate(attention_mask_np, axis=0)

                all_tokenized_np = np.concatenate(
                    [
                        oldid_np.astype(int).reshape(-1, 1),
                        input_ids_np,
                        attention_mask_np,
                    ],
                    axis=1,
                )  # (#data, 1 + length_constraint + length_constraint)
                np.save(filename, all_tokenized_np)


def load_data(args, return_docid2doc=True):
    # TODO need to be refactored with load_data_infer and glen_p2_dataset.py
    # TODO dataset directory should be cleaned up
    def process_func(
        doc_to_query_list,
        index,
        query,
        newid,
        rank=1,
        newid2doc=None,
        qryid2negnewids=None,
        docid_list=None,
    ):
        docid = "-".join([c for c in str(newid).split("-")])
        neg_docid_list, aug_query_list = [], []
        if args.aug_query:
            if args.aug_query_type == "aug_query":
                if newid in doc_to_query_list:
                    aug_query_list = doc_to_query_list[newid]
            else:
                for i in range(10):
                    aug_query_list.append(augment(query))

        return {
            "query": query,
            "docid": docid,
            "rank": rank,
            "neg_docid_list": neg_docid_list,
            "aug_query_list": aug_query_list,
        }

    result, doc_to_query_list, qryid2negnewids = None, None, None

    st_time = time()

    # Get original document
    newid2doc, oldid2doc, oldid2newid, newid2oldid = dict(), dict(), dict(), dict()
    if args.dataset_name == "nq320k":
        fname = f"data/nq320k/DOC_NQ_title+content.tsv"
        df = pd.read_csv(fname, sep="\t", dtype=str)
        if "query" in df.columns:
            df.rename(columns={"query": "doc_content"}, inplace=True)
        oldid2doc = dict(zip(df["oldid"], df["doc_content"]))

        print(f">>> CAUTION ~ Reading co-condenser version file ~")
        fname = f"data/nq320k/ID_NQ_{args.id_class}.tsv"
        df = pd.read_csv(fname, sep="\t", dtype=str).loc[:, ["oldid", args.id_class]]
        df.dropna(axis=0, inplace=True)

        if args.small_set in [1, 10]:
            small_id_df = pd.read_csv(
                f"data/nq320k/SMALL_NQ_{args.small_set}_oldid.tsv",
                sep="\t",
                dtype=str,
            )
            df = df[df["oldid"].isin(small_id_df["oldid"])]

        for [oldid, docid] in df.values.tolist():
            newid2doc[docid] = oldid2doc[oldid]
            oldid2newid[oldid] = docid
            newid2oldid[docid] = oldid
    elif args.dataset_name == "marco_passage":
        print(f"> Start reading original document and ids")
        st = time()
        fname = f"data/marco_passage/DOC_MARCO_collection.tsv"
        df = pd.read_csv(fname, encoding="utf-8", sep="\t", dtype=str).loc[
            :, ["oldid", "query"]
        ]
        oldid2doc = dict(zip(df["oldid"].values.tolist(), df["query"].values.tolist()))

        fname = f"data/marco_passage/ID_MARCO_{args.id_class}.tsv"
        df = pd.read_csv(fname, sep="\t", dtype=str).loc[:, ["oldid", args.id_class]]
        df.dropna(axis=0, inplace=True)
        print(f"> Reading original document and ids took {time()-st:.2f} secs")

        if args.small_set in [1]:
            small_id_df = pd.read_csv(
                f"data/marco_passage/SMALL_MARCO_{args.small_set}_oldid.tsv",
                sep="\t",
                dtype=str,
            )
            df = df[df["oldid"].isin(small_id_df["oldid"])]

        oldid2newid = dict(
            zip(df["oldid"].values.tolist(), df[args.id_class].values.tolist())
        )
        newid2oldid = dict(
            zip(df[args.id_class].values.tolist(), df["oldid"].values.tolist())
        )
        newid2doc = dict(
            zip(
                df[args.id_class].values.tolist(),
                [oldid2doc[oldid] for oldid in df["oldid"].values.tolist()],
            )
        )
        print(f"> Processing small document and ids took {time()-st:.2f} secs")

    ## load pre-defined id_class
    if "gtq" in args.query_type:
        print(f"> Reading gtq files")
        if args.dataset_name == "nq320k":
            fname = f"data/nq320k/GTQ_NQ_train.tsv"
        elif args.dataset_name == "marco_passage":
            fname = f"data/marco_passage/GTQ_MARCO_train.tsv"
        df = pd.read_csv(fname, encoding="utf-8", sep="\t", dtype=str)
        # insert newid and remove oldid
        df.insert(1, args.id_class, df["oldid"].map(oldid2newid))
        df = df.loc[:, ["query", args.id_class]]
        if args.small_set in [1, 10]:
            df.dropna(axis=0, inplace=True)
        assert not df.isnull().values.any()
        doc_to_query_list = defaultdict(set)
        for [query, docid] in df.loc[:, ["query", args.id_class]].values.tolist():
            doc_to_query_list[docid].add(query)
        docid_list = list(newid2doc.keys())
        result = tuple(
            process_func(
                doc_to_query_list,
                index,
                *row,
                newid2doc=newid2doc,
                qryid2negnewids=qryid2negnewids,
                docid_list=docid_list,
            )
            for index, row in enumerate(zip(df["query"], df[args.id_class]))
        )

    ## Query Generation Data
    if "qg" in args.query_type:
        if args.dataset_name == "nq320k":
            fname = f"data/nq320k/QG_NQ_top15.tsv"
        elif args.dataset_name == "marco_passage":
            fname = f"data/marco_passage/QG_MARCO_top1.tsv"
            if "qg20" in args.query_type:
                fname = "data/marco_passage/QG_MARCO_top20.tsv"
        df = pd.read_csv(fname, encoding="utf-8", sep="\t", dtype=str)
        # insert newid and remove oldid
        df.insert(1, args.id_class, df["oldid"].map(oldid2newid))
        df = df.loc[:, ["query", args.id_class]]
        df = df.dropna(axis=0)

        doc_to_query_list = defaultdict(set)
        for [query, docid] in df.loc[:, ["query", args.id_class]].values.tolist():
            doc_to_query_list[docid].add(query)
        temp = defaultdict(list)
        for k, v in doc_to_query_list.items():
            temp[k] = list(v)
        doc_to_query_list = temp

        docid_list = list(newid2doc.keys())
        result_add1 = tuple(
            process_func(
                doc_to_query_list,
                index,
                *row,
                newid2doc=newid2doc,
                qryid2negnewids=qryid2negnewids,
                docid_list=docid_list,
            )
            for index, row in enumerate(zip(df["query"], df[args.id_class]))
        )
        result = result_add1 if result is None else result + result_add1

    path_list = []
    ## Document first 64 terms
    if "doc" in args.query_type:
        if args.dataset_name == "nq320k":
            fname = f"data/nq320k/DOC_NQ_first64.tsv"
        elif args.dataset_name == "marco_passage":
            fname = f"data/marco_passage/DOC_MARCO_collection.tsv"
        path_list.append(fname)

    ## Document random chunk 64 terms
    if "aug" in args.query_type:
        if args.dataset_name == "nq320k":
            fname = f"data/nq320k/DOC_NQ_random64.tsv"
        elif args.dataset_name == "marco_passage":
            raise NotImplementedError("marco_passage document aug not available")
        path_list.append(fname)

    for file_path in path_list:
        print("file_path: ", file_path)
        df = pd.read_csv(file_path, encoding="utf-8", sep="\t", dtype=str)
        # insert newid and remove oldid
        df.insert(1, args.id_class, df["oldid"].map(oldid2newid))
        df = df.loc[:, ["query", args.id_class]]

        df.dropna(axis=0, inplace=True)
        assert not df.isnull().values.any()
        docid_list = list(newid2doc.keys())
        result_add1 = tuple(
            filter(
                None,
                (
                    process_func(
                        doc_to_query_list,
                        index,
                        *row,
                        newid2doc=newid2doc,
                        qryid2negnewids=qryid2negnewids,
                        docid_list=docid_list,
                    )
                    for index, row in enumerate(zip(df["query"], df[args.id_class]))
                ),
            )
        )
        result = result_add1 if result is None else result + result_add1

    print("&" * 20)
    print(result[0])
    print("&" * 20)

    print(f">> [Train] Loaded {len(result)} examples from {args.dataset_name} dataset.")
    print(f">> Processing {args.dataset_name} took {time()-st_time:.2f} secs")
    if return_docid2doc:
        return result, newid2doc, newid2oldid
    else:
        return result


def augment(query: str):
    if len(query) < 20 * 10:
        start_pos = np.random.randint(0, int(len(query) + 1 / 2))
        end_pos = np.random.randint(start_pos, len(query))
        span_length = max(start_pos - end_pos, 10 * 10)
        new_query = str(query[start_pos : start_pos + span_length])
    else:
        start_pos = np.random.randint(0, len(query) - 10 * 10)
        end_pos = np.random.randint(start_pos + 5 * 10, len(query))
        span_length = min(start_pos - end_pos, 20 * 10)
        new_query = str(query[start_pos : start_pos + span_length])
    return new_query
