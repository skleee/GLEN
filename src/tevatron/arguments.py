from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class GLENTrainingArguments(TrainingArguments):
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    warmup_ratio: float = field(default=0.0)
    negatives_x_device: bool = field(
        default=False, metadata={"help": "share negatives across devices"}
    )
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    project_name: Optional[str] = field(
        default="EMNLP2023", metadata={"help": "wandb project name"}
    )
    wandb_tag: Optional[str] = field(default=None, metadata={"help": "wandb tag"})
    save_steps: int = field(
        default=20000, metadata={"help": "save model every x steps"}
    )
    save_strategy: str = field(default="steps", metadata={"help": "save strategy"})
    save_total_limit: int = field(default=5, metadata={"help": "save total limit"})
    res1_save_path: str = field(default="")
    val_check_interval: float = field(
        default=0.2, metadata={"help": "validation check interval for each epoch"}
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "evaluation strategy"}
    )


@dataclass
class GLENP1TrainingArguments(GLENTrainingArguments):
    metric_for_best_model: str = field(
        default="eval_recall@1", metadata={"help": "metric for best model"}
    )
    num_train_epochs: float = field(
        default=500.0, metadata={"help": "number of training epochs"}
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "adam epsilon"})
    warmup_steps: int = field(default=0, metadata={"help": "warmup steps"})
    weight_decay: float = field(default=1e-4, metadata={"help": "weight decay"})
    learning_rate: float = field(default=2e-4, metadata={"help": "learning rate"})
    decoder_learning_rate: float = field(
        default=1e-4, metadata={"help": "decoder learning rate"}
    )


@dataclass
class GLENP2TrainingArguments(GLENTrainingArguments):
    learning_rate: float = field(default=5e-5, metadata={"help": "learning rate"})
    grad_cache: bool = field(
        default=False, metadata={"help": "Use gradient cache update"}
    )
    gc_q_chunk_size: int = field(default=128)
    gc_p_chunk_size: int = field(default=128)


@dataclass
class GLENDataArguments:
    dataset_name: str = field(
        default=None,
        metadata={"help": "huggingface dataset name or custom dataset name"},
    )
    encode_train_qry: bool = field(default=False)
    test100: int = field(
        default=0,
        metadata={"help": "Debug mode. Only use a subset of the data (100 examples)"},
    )

    query_type: str = field(
        default="gtq_doc_aug_qg",
        metadata={
            "help": "gtq: ground turth query, qg: generated query, doc: just use top64 doc tokens, aug: use random doc token"
        },
    )
    small_set: int = field(
        default=0, metadata={"help": "nq320k small set size", "choices": [0, 1, 10]}
    )
    aug_query: bool = field(
        default=True, metadata={"help": "whether to use augmented query"}
    )
    aug_query_type: str = field(
        default="corrupted_query",
        metadata={
            "help": "augmented query type",
            "choices": ["corrupted_query", "aug_query"],
        },
    )
    id_class: str = field(
        default="t5_bm25_truncate_3", metadata={"help": "id class for nq320k"}
    )


@dataclass
class GLENP1DataArguments(GLENDataArguments):
    max_input_length: int = field(default=156, metadata={"help": "max input length"})


@dataclass
class GLENP2DataArguments(GLENDataArguments):
    max_input_length: int = field(
        default=156, metadata={"help": "max input length used for making id"}
    )
    train_n_passages: int = field(default=0)
    positive_passage_no_shuffle: bool = field(
        default=True, metadata={"help": "always use the first positive passage"}
    )
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )
    negative_passage_type: str = field(
        default="self",
        metadata={
            "help": "ibn: in batch negative, hard: hard negative, random: random negative",
            "choices": ["random", "self"],
        },
    )
    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=156,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class GLENModelArguments:
    model_name_or_path: str = field(
        default="t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    num_layers: int = field(default=12)
    num_decoder_layers: int = field(default=12)
    d_ff: int = field(default=3072)
    d_model: int = field(default=768)
    num_heads: int = field(default=12)
    d_kv: int = field(default=64)
    use_past_key_values: bool = field(default=True)
    load_pretrained_st5_checkpoint: str = field(default=None)
    mask_special_tokens_for_decoding: bool = field(default=True)
    tie_decode_embeddings: bool = field(default=True)
    tie_word_embeddings: bool = field(default=True)
    dropout_rate: float = field(default=0.1)

    # Inference Arguments
    length_penalty: float = field(default=0.8)
    num_return_sequences: int = field(
        default=100, metadata={"help": "number of return sequences."}
    )
    early_stopping: bool = field(default=False)
    tree: int = field(default=1)
    reranking: str = field(
        default="cosine",
        metadata={
            "help": "random, cosine, mse",
            "choices": ["random", "cosine", "mse"],
        },
    )

    gen_method: str = field(
        default="greedy", metadata={"help": "Only used when decoder_input is docid"}
    )  # greedy, beam_search, top_k, top_p
    infer_ckpt: str = field(
        default="",
        metadata={
            "help": "Path to checkpoint file (e.g., logs/GLEN-6700/pytorch_model.bin). Model args will not be loaded from model_args.json"
        },
    )
    infer_dir: str = field(
        default="",
        metadata={
            "help": "Path to directory that contains .bin files (e.g., logs/GLEN-6700)"
        },
    )
    logs_dir: str = field(
        default="logs", metadata={"help": "Path to save inference results"}
    )
    docid_file_name: str = field(default="")

    max_output_length: int = field(
        default=5,
        metadata={"help": "max output length. For phase 2, used only for inference"},
    )


@dataclass
class GLENP1ModelArguments(GLENModelArguments):
    verbose_valid_query: int = field(
        default=1,
        metadata={
            "help": "0: no verbose, 1: verbose with 10^1 queries, 2: verbose with all queries",
            "choices": [0, 1, 2],
        },
    )

    freeze_encoder: bool = field(default=False)
    freeze_embeds: bool = field(default=False)
    pretrain_encoder: bool = field(default=True)
    pretrain_decoder: bool = field(default=True)
    output_vocab_size: int = field(default=10)

    Rdrop: float = field(default=0.15)
    input_dropout: int = field(default=1)
    decoder_input: str = field(default="doc_rep")  # doc_rep, doc_id


@dataclass
class GLENP2ModelArguments(GLENModelArguments):
    softmax_temperature: float = field(default=1.0)
    num_multi_vectors: int = field(default=3)
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"},
    )

    infonce_loss: float = field(default=1.0)  #  pairwise ranking loss
    q_to_docid_loss: float = field(default=0.5)  # pointwise retrieval loss (first term)
    cosine_point_loss: float = field(
        default=0.25
    )  # pointwise retrieval loss (second term)
    do_docid_temperature_annealing: bool = field(default=True)
    docid_temperature: float = field(default=1.0)
    docid_temperature_min: float = field(default=1e-5)
