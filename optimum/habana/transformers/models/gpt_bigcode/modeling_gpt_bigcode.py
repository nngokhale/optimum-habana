# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeAttention,
    GPTBigCodeForCausalLM,
    upcast_masked_softmax,
    upcast_softmax,
)

from ...modeling_attn_mask_utils import GaudiAttentionMaskConverter


try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

import habana_frameworks.torch.core as htcore


#  FusedScaledDotProductAttention
class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        self._hpu_kernel_fsdpa = fusedSDPA

    def forward(self, query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode, enable_recompute):
        return self._hpu_kernel_fsdpa.apply(
            query, key, value, attn_mask, dropout_p, is_casual, scale, softmax_mode, enable_recompute
        )


class GaudiGPTBigCodeAttention(GPTBigCodeAttention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.fused_scaled_dot_product_attention = ModuleFusedSDPA(FusedSDPA) if FusedSDPA is not None else None
        self.block_size = 4096

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """
        This method should be deleted when https://github.com/huggingface/transformers/pull/34508 is merged.
        Copied from GPTBigCodeAttention._attn: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
        The only differences are:
        - in self._attn, use torch.matmul instead of torch.baddbmm when the device used for query is not cpu
        """
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights = torch.zeros_like(attn_weights)
            attn_weights = torch.baddbmm(attn_weights, query, key, beta=1, alpha=scale_factor).view(attn_shape)
        else:
            # Formula for torch.baddbmm: out = beta * attn_weights + scale_factor * (query ⋅ key)
            # for beta = 0, it simplifies to: out = scale_factor * (query ⋅ key)
            attn_weights = (torch.matmul(query, key) * scale_factor).view(attn_shape)

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)
        else:
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def gaudi_flash_attn_v1(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        dropout_rate,
        is_causal,
        scale,
        softmax_mode,
        enable_recompute,
        q_block_size,
    ):
        """
        Gaudi version of Flash Attention V1 to support long sequence at prompt phase
        Causal mask is not supported in this optimization
        """
        if is_causal:
            raise ValueError("Causal mask is not supported for long input sequences")

        q_len = query_layer.size(-2)
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        q_padding = q_tiles * q_block_size - q_len
        query_layer = F.pad(query_layer, (0, 0, 0, q_padding), "constant", 0)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 0, 0, q_padding), "constant", -10000.0)
        row_o_list = []
        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = query_layer[:, :, s:e, :]
            row_mask = attention_mask[:, :, s:e, :]
            attn_output_partial = self.fused_scaled_dot_product_attention(
                row_q, key_layer, value_layer, row_mask, dropout_rate, is_causal, scale, softmax_mode, enable_recompute
            )
            row_o_list.append(attn_output_partial)
        attn_output = torch.cat(row_o_list, dim=-2)
        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]
        return attn_output

    def apply_FusedSDPA(
        self,
        query,
        key,
        value,
        attention_mask=None,
        flash_attention_recompute=False,
        flash_attention_fast_softmax=False,
        flash_attention_causal_mask=False,
    ):
        """
        Copied from GPTBigCodeSdpaAttention._attn: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
        The only differences are:
        - replaced torch.nn.functional.scaled_dot_product_attention with Habana's FusedSDPA
        - removed WA for key and value tensor expanding over heads dimension. That WA also works but dramatically drops throughput
        - added args use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask to control parameters of FusedSDPA
        - added special case handling for input larger 8192 with function gaudi_flash_attn_v1
        """

        scale = None
        if not self.scale_attn_weights:
            scale = 1

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]

        if self.multi_query:
            query_length = query_shape[1]

            # SDPA requires the dimension [..., sequence_length, head_dim].
            query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)

            # Without these unsqueeze, SDPA complains as the query and key/value have a different number of dimensions.
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        else:
            query_length = query_shape[-1]

            if attention_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

        sdpa_result = None
        enable_recompute = flash_attention_recompute and query_length > 1

        if query_length > 1 and flash_attention_causal_mask:
            attention_mask = None
            use_causal_mask = True
        else:
            use_causal_mask = self.is_causal and attention_mask is None and query_length > 1

        if query_length > 8192:
            sdpa_result = self.gaudi_flash_attn_v1(
                query,
                key,
                value,
                attention_mask,
                self.attn_pdrop if self.training else 0.0,
                use_causal_mask,
                scale,
                "fast" if flash_attention_fast_softmax else "None",
                enable_recompute,
                self.block_size,
            )
            htcore.mark_step()
        else:
            sdpa_result = self.fused_scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask,
                self.attn_pdrop if self.training else 0.0,
                use_causal_mask,
                scale,
                "fast" if flash_attention_fast_softmax else "None",
                enable_recompute,
            )

        if self.multi_query:
            # (batch_size, num_heads, seq_len, head_dim) --> (batch_size, seq_len, num_heads, head_dim)
            sdpa_result = sdpa_result.transpose(1, 2)

            # Reshape is kind of expensive here, as it does a memory copy,
            # but I did not manage to make away without it (logits do not match when using view)
            # (batch_size, seq_len, num_heads, head_dim) --> (batch_size, seq_len, num_heads * head_dim)
            sdpa_result = sdpa_result.reshape(query_shape)

        return sdpa_result, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: Optional[int] = None,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        """
        Copied from GPTBigCodeAttention.forward: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
        The only differences are:
        - add new args token_idx, use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask
        - optimize KV cache
        """
        if use_flash_attention:
            assert self.fused_scaled_dot_product_attention is not None, (
                "Can't load HPU fused scaled dot-product attention kernel. Please retry without flash attention"
            )

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        _, q_len, _ = hidden_states.size()
        bucket_internal_decode_stage = cache_idx is not None and q_len == 1

        if not bucket_internal_decode_stage:
            if layer_past is not None:
                past_key, past_value = layer_past.split((self.head_dim, self.head_dim), dim=-1)
                if token_idx is not None:
                    # Using out of place version of index_add_() to ensure the intermediate tensors are not lost when HPU graphs are enabled.
                    key = past_key.index_add(1, token_idx - 1, key - torch.index_select(past_key, 1, token_idx - 1))
                    value = past_value.index_add(
                        1, token_idx - 1, value - torch.index_select(past_value, 1, token_idx - 1)
                    )
                else:
                    key = torch.cat((past_key, key), dim=-2)
                    value = torch.cat((past_value, value), dim=-2)
            present = torch.cat((key, value), dim=-1) if use_cache else None
        else:
            assert token_idx is not None, "Invalid parameters: token_idx is None at decode stage with bucket_internal"
            assert layer_past is not None, (
                "Invalid parameters: layer_past is None at decode stage with bucket_internal"
            )

            past_key, past_value = layer_past.split((self.head_dim, self.head_dim), dim=-1)
            key = past_key.index_copy_(1, token_idx - 1, key)
            value = past_value.index_copy_(1, token_idx - 1, value)
            present = layer_past

        if bucket_internal_decode_stage:
            key = key[:, :cache_idx, :]
            value = value[:, :cache_idx, :]
            attention_mask = attention_mask[:, :, :, :cache_idx]

        if not output_attentions and head_mask is None and use_flash_attention:
            # Difference with the original implementation: there is no need to transpose the key here,
            # as SDPA expects seq_length to be at index -2 for the key as well
            attn_output, attn_weights = self.apply_FusedSDPA(
                query,
                key,
                value,
                attention_mask,
                flash_attention_recompute,
                flash_attention_fast_softmax,
                flash_attention_causal_mask,
            )
        else:
            attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        if bucket_internal_decode_stage:
            # Return only past key value shapes and not the tensors during decode phase (q len is 1)
            # to avoid making past key values as persistent output tensors of HPU graphs.
            present = present.shape

        outputs = (attn_output, present)
        if output_attentions:
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def gaudi_gpt_bigcode_block_forward(
    self,
    hidden_states: Optional[Tuple[torch.Tensor]],
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    token_idx: Optional[torch.Tensor] = None,
    use_flash_attention: Optional[bool] = False,
    flash_attention_recompute: Optional[bool] = False,
    flash_attention_fast_softmax: Optional[bool] = False,
    flash_attention_causal_mask: Optional[bool] = False,
    cache_idx: Optional[int] = None,
    **kwargs,
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Copied from GPTBigCodeBlock.forward: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    The only differences are:
    - add new args token_idx, use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask
    """
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        token_idx=token_idx,
        use_flash_attention=use_flash_attention,
        flash_attention_recompute=flash_attention_recompute,
        flash_attention_fast_softmax=flash_attention_fast_softmax,
        flash_attention_causal_mask=flash_attention_causal_mask,
        cache_idx=cache_idx,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    return outputs  # hidden_states, present, (attentions, cross_attentions)


def gaudi_gpt_bigcode_model_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    token_idx: Optional[torch.Tensor] = None,
    use_flash_attention: Optional[bool] = False,
    flash_attention_recompute: Optional[bool] = False,
    flash_attention_fast_softmax: Optional[bool] = False,
    flash_attention_causal_mask: Optional[bool] = False,
    cache_idx: Optional[int] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    """
    Copied from GPTBigCodeModel.forward: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    The only differences are:
    - add new args token_idx, use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask
    - if token_idx and past_key_values are passed, set self_attention_mask based on the static shape of past_key_values
    """

    # This flag used for correct tensors reshape for attention kernel
    self._use_sdpa = use_flash_attention

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0].size(-2)

    if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, past_length : input_shape[-1] + past_length :]
    elif position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)

    # Self-attention mask.
    query_length = input_shape[-1]
    key_length = past_length + query_length
    if past_length > 0 and token_idx is not None:
        self_attention_mask = self.bias[None, past_length - 1 : past_length, :past_length]
    else:
        self_attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

    if attention_mask is not None:
        self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
            dtype=torch.bool, device=self_attention_mask.device
        )

    # MQA models: (batch_size, query_length, n_heads, key_length)
    # MHA models: (batch_size, n_heads, query_length, key_length)
    self_attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)

    if self._use_sdpa and head_mask is None and not output_attentions:
        # SDPA with a custom mask is much faster in fp16/fp32 dtype rather than bool. Cast here to floating point instead of at every layer.
        dtype = self.wte.weight.dtype
        min_dtype = torch.finfo(dtype).min
        self_attention_mask = torch.where(
            self_attention_mask,
            torch.full([], 0.0, dtype=dtype, device=self_attention_mask.device),
            torch.full([], min_dtype, dtype=dtype, device=self_attention_mask.device),
        )

        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        if self.multi_query:
            # gpt_bigcode using MQA has the bad taste to use a causal mask with shape
            # [batch_size, target_length, 1, source_length], not compatible with SDPA, hence this transpose.
            self_attention_mask = self_attention_mask.transpose(1, 2)

        if query_length > 1 and attention_mask is not None:
            # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
            # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
            self_attention_mask = GaudiAttentionMaskConverter._unmask_unattended(
                self_attention_mask, min_dtype=min_dtype
            )

    attention_mask = self_attention_mask

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.add_cross_attention and encoder_hidden_states is not None and encoder_attention_mask is not None:
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask.unsqueeze(1)
        assert encoder_attention_mask.dim() == 3
        encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(2 if self.multi_query else 1)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = [] if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
                None,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                token_idx=token_idx,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                flash_attention_fast_softmax=flash_attention_fast_softmax,
                flash_attention_causal_mask=flash_attention_causal_mask,
                cache_idx=cache_idx,
            )

        hidden_states = outputs[0]
        if use_cache:
            presents.append(outputs[1])

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


class GaudiGPTBigCodeForCausalLM(GPTBigCodeForCausalLM):
    """
    Inherits from GPTBigCodeForCausalLM: https://github.com/huggingface/transformers/blob/v4.40-release/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
    The only differences are:
    - add new args token_idx, use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask
    - add token_idx, use_flash_attention, flash_attention_recompute, flash_attention_fast_softmax, flash_attention_causal_mask into model_inputs
    - when KV cache is enabled, slice next_input_ids from input_ids based on the token_idx
    - when KV cache is enabled, slice next_position_ids from position_ids based on the token_idx
    - support for internal bucketing
    """

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, token_idx=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        bucket_internal = kwargs.get("bucket_internal", False)
        # Omit tokens covered by past_key_values
        if past_key_values:
            if token_idx is not None:
                idx = token_idx + kwargs.get("inputs_embeds_offset", 0) - 1
                input_ids = torch.index_select(input_ids, 1, idx)
                if token_type_ids is not None:
                    token_type_ids = torch.index_select(token_type_ids, 1, token_idx - 1)
            else:
                if self.config.multi_query:
                    past_length = past_key_values[0].shape[1]
                else:
                    past_length = past_key_values[0].shape[2]

                # Some generation methods already pass only the last input ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if past_key_values is None and bucket_internal and token_idx is not None:
            # KV cache will be padded with bucket internal hence for the 1st token we can slice the inputs till token idx for the fwd pass.
            input_ids = input_ids[:, :token_idx]
            attention_mask = attention_mask[:, :token_idx]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                if token_idx is not None:
                    position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                else:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "token_idx": token_idx,
                "use_flash_attention": kwargs.get("use_flash_attention", False),
                "flash_attention_recompute": kwargs.get("flash_attention_recompute", False),
                "flash_attention_fast_softmax": kwargs.get("flash_attention_fast_softmax", False),
                "flash_attention_causal_mask": kwargs.get("flash_attention_causal_mask", False),
                "cache_idx": kwargs.get("cache_idx", None),
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        flash_attention_fast_softmax: Optional[bool] = False,
        flash_attention_causal_mask: Optional[bool] = False,
        cache_idx: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
            flash_attention_fast_softmax=flash_attention_fast_softmax,
            flash_attention_causal_mask=flash_attention_causal_mask,
            cache_idx=cache_idx,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
