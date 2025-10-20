import torch, torch.nn as nn
import math
from .whisper import VariableLengthWhisperForAudioClassification, VariableLengthWhisperEncoder
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Union
from transformers.modeling_outputs import SequenceClassifierOutput

import numpy as np

class AudioProjector(nn.Module):
    def __init__(self, d_in, d_out, use_pe=True):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.use_pe = use_pe
        if use_pe:
            self.register_buffer("pe_cache", torch.empty(0), persistent=False)

    def sinusoidal_pe(self, T, d, dtype=torch.float32):
        # standard transformer PE
        pe = torch.zeros(T, d, dtype=dtype)
        position = torch.arange(0, T, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: (B, T_a, D_in)
        z = self.proj(x)  # (B, T_a, D_out)
        if self.use_pe:
            B, T, D = z.shape
            if self.pe_cache.shape[:2] != (T, D):
                self.pe_cache = self.sinusoidal_pe(T, D, dtype=z.dtype).to(z.device)
            z = z + self.pe_cache.unsqueeze(0)  # broadcast to batch
        return z

from torch import nn
from transformers import LlamaModel, LlamaConfig, WhisperConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import copy

class WhisperSmolLMClassifierConfig(PretrainedConfig):
    is_composition = True
    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)
        llama_config = kwargs.pop("llama", {})
        whisper_config = kwargs.pop("whisper", {})

        self.llama = LlamaConfig(**llama_config)
        self.whisper = WhisperConfig(**whisper_config)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["llama"] = self.llama.to_dict()
        output["whisper"] = self.whisper.to_dict()
        return output

class WhisperSmolLMClassifier(PreTrainedModel):
    config_class = WhisperSmolLMClassifierConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.audio_encoder = VariableLengthWhisperEncoder(config.whisper)
        self.llm = LlamaModel(config.llama)
        self.audio_projector = AudioProjector(self.audio_encoder.config.d_model, self.llm.config.hidden_size, use_pe=False)
        self.score = nn.Linear(self.llm.config.hidden_size, 2, bias=False)

    def build_inputs(self, audio_features, input_ids):
        aud_h = self.audio_encoder(audio_features).last_hidden_state
        aud_h = self.audio_projector(aud_h)
        text_h = self.llm.embed_tokens(input_ids)

        inputs_embeds = torch.cat([aud_h, text_h], dim=1)

        return inputs_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        audio_features,
        **kwargs,
    ):
        input_embeds = self.build_inputs(audio_features, input_ids)
        res = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask)

        hidden_states = res.last_hidden_state
        logits = self.score(hidden_states)
        batch_size = input_ids.shape[0]

        last_non_pad_token = -1
        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.llm.config.pad_token_id is None:
        #     last_non_pad_token = -1
        # elif input_ids is not None:
        #     # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
        non_pad_mask = (input_ids != self.llm.config.pad_token_id).to(logits.device, torch.int32)
        token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        last_non_pad_token = last_non_pad_token + (input_embeds.shape[1] - input_ids.shape[-1])

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        return SequenceClassifierOutputWithPast(
            logits=pooled_logits,
            past_key_values=res.past_key_values,
            hidden_states=res.hidden_states,
            attentions=res.attentions,
        )
