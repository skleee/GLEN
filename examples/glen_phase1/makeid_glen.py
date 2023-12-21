import json
import logging
import os
import sys
import torch
import warnings

import numpy as np

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
        raise NotImplementedError("Multi-GPU make_id is not supported.")

    if os.path.exists(os.path.join(model_args.infer_dir, "model_args.json")):
        print(
            f"> Load model arguments from {os.path.join(model_args.infer_dir, 'model_args.json')}"
        )
        with open(os.path.join(model_args.infer_dir, "model_args.json"), "r") as f:
            model_args_dict = json.load(f)
        model_args = ModelArguments(**model_args_dict)
    else:
        print(f"> Not found model arguments from {os.path.join(model_args.infer_dir)}")

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

    # Custom dataset: NQ320k, MS MARCO Passage, nfcorpus, arguana
    if data_args.dataset_name in ["nq320k", "marco_passage", "nfcorpus", "arguana"]:
        encode_dataset = GLENP1EncodeDataset(
            data_args=data_args,
            tokenizer=tokenizer,
            max_len=data_args.max_input_length,
            task="make_id",
        )
    else:
        raise NotImplementedError(f"{data_args.dataset_name} is not supported")

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = model.to(training_args.device)
    model.eval()

    model.tokenizer = tokenizer
    if model_args.mask_special_tokens_for_decoding:
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

    max_output_length = model_args.max_output_length

    all_ids = []
    decoder_attention_mask = torch.ones((1, max_output_length), dtype=torch.long).cuda()
    for batch in tqdm(encode_loader, dynamic_ncols=True, desc="make id"):
        with torch.no_grad():
            if model_args.decoder_input == "doc_id":
                model_args.num_return_sequences = 1
                outs = model.hf_model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=False,
                    decoder_attention_mask=decoder_attention_mask,
                    max_length=max_output_length,
                    num_beams=model_args.num_return_sequences,
                    length_penalty=model_args.length_penalty,
                    num_return_sequences=model_args.num_return_sequences,
                    early_stopping=model_args.early_stopping,
                    decode_vocab_size=decode_vocab_size,
                )
                dec = []
                for ids in outs[:, 1:]:
                    dec.append(
                        "<->".join(
                            tokenizer.convert_ids_to_tokens(
                                ids, skip_special_tokens=False
                            )
                        )
                    )
                out_logits = np.zeros_like(outs.cpu().numpy())

            elif model_args.decoder_input == "doc_rep":
                past_key_values, encoder_outputs = None, None
                decoder_inputs_embeds = model.hf_model.get_input_embeddings()(
                    torch.tensor([0], dtype=torch.long, device=torch.device("cuda"))
                )  # [1, H]
                decoder_inputs_embeds = decoder_inputs_embeds.unsqueeze(0).repeat(
                    batch["source_ids"].shape[0], 1, 1
                )  # [B, 1, H]
                decoder_attention_mask_full = torch.ones(
                    batch["source_ids"].shape[0],
                    max_output_length - 1,
                    dtype=torch.long,
                    device=torch.device("cuda"),
                )
                outs, out_logits = [], []
                for i in range(max_output_length - 1):
                    decoder_attention_mask = decoder_attention_mask_full[:, : i + 1]
                    psg_out = model.hf_model(
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
                    if model_args.mask_special_tokens_for_decoding:
                        psg_out.logits[:, :, special_token_ids] = -float("inf")
                    out = psg_out.logits[:, -1, :].argmax(dim=-1)  # [B, L-1]
                    out_logit = (
                        psg_out.logits[:, -1, :]
                        .gather(1, out.unsqueeze(-1))
                        .squeeze(-1)
                    )
                    outs.append(out.cpu().numpy())
                    out_logits.append(out_logit.cpu().detach().numpy())
                outs = np.stack(outs, axis=1)
                out_logits = np.stack(out_logits, axis=1)

                dec = []
                for ids in outs:
                    dec.append(
                        "<->".join(
                            tokenizer.convert_ids_to_tokens(
                                ids, skip_special_tokens=False
                            )
                        )
                    )

            else:
                raise NotImplementedError(
                    f"{model_args.decoder_input} is not supported"
                )

        texts = []
        for ids in batch["source_ids"]:
            texts.append(tokenizer.decode(ids.numpy(), skip_special_tokens=True))
        oldids = []
        for oldid in batch["oldids"]:
            oldids.append(oldid)

        out_logits = np.round(out_logits.astype(np.float64), 4)
        for oldid, text, pred, out_logit in zip(
            oldids, texts, dec, out_logits.tolist()
        ):
            out_logit = "<->".join([str(x) for x in out_logit])
            all_ids.append([oldid, pred, out_logit, text])

    if model_args.docid_file_name == "":
        docid_file_name = f"{model.__class__.__name__}_len_{data_args.max_input_length}_{data_args.dataset_name}"

    docid_file_name = (
        "/".join(model_args.infer_dir.split("/")[:-1])
        + "/"
        + model_args.docid_file_name
        + ".tsv"
    )
    with open(docid_file_name, "w") as f:
        for oldid, pred, out_logit, text in all_ids:
            f.write(f"{oldid}\t{pred}\t{out_logit}\t{text}\n")
    print(f"> docid file is saved to {docid_file_name}")


if __name__ == "__main__":
    main()
