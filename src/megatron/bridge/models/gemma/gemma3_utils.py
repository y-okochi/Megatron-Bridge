# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from functools import lru_cache
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from megatron.core.transformer import ModuleSpec, TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import ModuleSpec, TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.utils import safe_import_from

TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TELayerNormColumnParallelLinear, _ = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear"
)
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)

def gemma3_layer_spec(config) -> ModuleSpec:
    """Gemma3 custom layer spec."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma3TEDotProductAttention,  # mixed gloabl/local attn
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # residual link
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,  # residual link
        ),
    )

class Gemma3SelfAttention(SelfAttention):
    """Gemma3 self attention.

    Uses local rope embedding for local layers,
    global rope embedding for global layers.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Switch to either local or global rope embedding before forward"""
        assert isinstance(rotary_pos_emb, tuple)
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern):
            final_rotary_pos_emb = rotary_pos_emb[0]
        else:
            final_rotary_pos_emb = rotary_pos_emb[1]
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=final_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )


class Gemma3TEDotProductAttention(TEDotProductAttention):
    """Gemma3 core attention.

    Switches between global and local sliding window attention
    based on the layer_number and pre-defined layer pattern.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        # Overwrite config.window_size based on layer_number
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # local attention, (q, k)
            config.window_size = (config.window_size, 0)
        else:
            # global attention
            config.window_size = None

        # The VL model calculates mask manually
        if config.is_vision_language:
            attn_mask_type = AttnMaskType.arbitrary

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )

class Gemma3LanguageModelEmbedding(LanguageModelEmbedding):
    """Gemma3 language token embedding.

    Adds a normalization to the embedding.
    """

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Calculate embedding and normalize"""
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        embeddings = embeddings * (self.config.hidden_size**0.5)
        return embeddings


class Gemma3RotaryEmbedding(RotaryEmbedding):
    """Gemma3 position rope embedding.

    Calculates rope embeddings for both local and global attention layers.
    """

    def __init__(
        self,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        **kwargs,
    ):
        # The rope scaling in RotaryEmbedding is not linear scaling,
        # so this flag must be off. Will calculate linear scaling below.
        assert rope_scaling is False

        # Get inv_freq for global attention layers
        super().__init__(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base,
            **kwargs,
        )
        self.inv_freq /= rope_scaling_factor

        # Setup Rotary Embedding for local attentions
        self.rope_local = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base_local,
            **kwargs,
        )

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Get global and local rope embedding"""
        rope_global = super().forward(max_seq_len, offset, packed_seq)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq)
        return rope_local, rope_global


def _is_local_attn_layer(
    layer_number: int,
    layer_pattern: Tuple[int, int],
) -> bool:
    pattern_size = sum(layer_pattern)
    return layer_number % pattern_size != 0

class TERowParallelLinearLayerNorm(TERowParallelLinear):
    """Modified From TERowParallelLinear with an additional Post-LN."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x):
        """Forward with additional Post LN on output"""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias
