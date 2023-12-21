import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import logging
from torch.nn import CrossEntropyLoss

from .glen_t5_modeling import T5Stack, T5PreTrainedModel
from .glen_t5_outputs import Seq2SeqLMOutput

logger = logging.get_logger(__name__)


class T5ForConditionalGeneration_GLEN(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        if "Rdrop" in config.__dict__:
            self.Rdrop = config.Rdrop
        self.Rdrop_loss = "Contrast"

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decode_vocab_size = getattr(config, "decode_vocab_size", None)
        tie_decode_embedding = getattr(config, "tie_decode_embedding", None)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", None)
        self.max_output_length = getattr(config, "max_output_length", None)

        if tie_word_embeddings:
            self.decode_embeddings = self.shared
            config.decode_vocab_size = config.vocab_size
        else:
            self.decode_embeddings = nn.Embedding(
                self.decode_vocab_size, config.d_model
            )

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.decode_embeddings)

        self.lm_head = nn.Linear(
            config.d_model, self.decode_vocab_size, bias=False
        )  # [decode_vocab_size, emb_dim]

        if tie_decode_embedding:
            self._tie_or_clone_weights(self.lm_head, self.decode_embeddings)

        # for generation
        config.vocab_size = config.decode_vocab_size

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        input_mask=None,
        logit_mask=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        lm_weights=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        decoder_index=-1,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert (
                labels is None
            ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        var_list = [
            decoder_input_ids,
            decoder_attention_mask,
            decoder_inputs_embeds,
            past_key_values,
            hidden_states,
            attention_mask,
            head_mask,
            use_cache,
            output_attentions,
            output_hidden_states,
        ]

        self_decoder = self.decoder
        self_lm_head = self.lm_head

        decoder_outputs = self_decoder(
            input_ids=var_list[0],
            attention_mask=var_list[1],
            inputs_embeds=var_list[2],
            past_key_values=var_list[3],
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=var_list[5],
            head_mask=var_list[6],
            use_cache=var_list[7],
            output_attentions=var_list[8],
            output_hidden_states=var_list[9],
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (
            self.model_dim**-0.5
        )  # shape(batch_size, sequence_length, config.d_model)

        lm_logits = self_lm_head(
            sequence_output
        )  # shape(batch_size, sequence_length, config.vocab_size)

        if not self.training and labels is not None:
            orig_loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            if self.Rdrop > 0 and self.training:
                bz = lm_logits.shape[0]
                sl = lm_logits.shape[1]

                # Rdrop contrastive loss
                neg_logits_1 = sequence_output.transpose(0, 1)  # [sl, bz, vocab_size]
                neg_logits_2 = neg_logits_1.transpose(1, 2)  # [sl, vocab_size, bz]
                neg_logits = torch.bmm(
                    neg_logits_1, neg_logits_2
                )  # [sl, bz, bz_logits]
                neg_mask = -1e9 * torch.eye(bz).to(neg_logits.device)
                neg_logits = neg_logits + neg_mask.unsqueeze(0)
                neg_logits = F.softmax(
                    neg_logits.view(-1, bz), dim=-1
                )  # [sl*bz, bz_logits]
                contrast_labels = torch.cat(
                    [torch.arange(bz // 2, bz), torch.arange(0, bz // 2)], dim=-1
                )
                contrast_labels = contrast_labels.to(neg_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(sl, 1).view(-1)
                dist_loss = loss_fct(neg_logits, contrast_labels)

                loss = orig_loss + self.Rdrop * dist_loss
            else:
                loss = orig_loss
        else:
            loss = None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return_result = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return_result.labels = labels
        if self.training and self.Rdrop > 0 and labels is not None:
            return_result.orig_loss = orig_loss
            return_result.dist_loss = self.Rdrop * dist_loss

        return_result.encoder_outputs = encoder_outputs
        return_result.lm_logits = lm_logits
        return return_result

    def prepare_inputs_for_generation(
        self, input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past
