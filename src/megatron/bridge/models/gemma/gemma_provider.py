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

from dataclasses import dataclass, field
from typing import Callable, Union

import torch
import math

from megatron.core.transformer import ModuleSpec
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.gemma.gemma3_utils import openai_gelu, gemma3_layer_spec, Gemma3LanguageModelEmbedding, Gemma3RotaryEmbedding
from megatron.bridge.utils import fusions

@dataclass
class GemmaModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Gemma3 models."""
    seq_length: int = 131_072

    # embedding
    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)  # (local, global)
    share_embeddings_and_output_weights: bool = True

    # norm
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # attention
    window_size: tuple = 512  # local
    interleaved_attn_pattern: tuple = (5, 1)  # (local, global)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rope_scaling_factor: float = 1.0
    # Disable cuDNN attention since TE 1.8 does not support head dim > 128
    attention_backend: AttnBackend = AttnBackend.flash

    # mlp
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = field(default_factory=lambda: openai_gelu)

    # Do not change
    is_vision_language: bool = False
    flash_decode: bool = False
    gradient_accumulation_fusion: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["GemmaModelProvider"], ModuleSpec]] = field(default_factory=lambda: gemma3_layer_spec)
    scatter_embedding_sequence_parallel: bool = True
    apply_rope_fusion: bool = field(default_factory=fusions.can_enable_apply_rope_fusion)
    masked_softmax_fusion: bool = field(default_factory=fusions.can_enable_masked_softmax_fusion)

    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma3 model.

        Replaces the model's embedding and rope with customized Gemma3 ones.

        Args:
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        rotary_base_local, rotary_base_global = self.rotary_base
        # Trick megatron's RotaryEmbedding to initialize the model successfully
        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)
        # Replace model's embedding and rope with customized ones
        if hasattr(model, 'embedding'):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )
        model.rotary_pos_emb = Gemma3RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            rope_scaling_factor=self.rope_scaling_factor,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
        )
        if hasattr(model, 'embedding') or hasattr(model, 'output_layer'):
            model.setup_embeddings_and_output_layer()
        return model

@dataclass
class Gemma3ModelProvider1B(GemmaModelProvider):
    """Gemma3 1B config"""

    is_vision_language: bool = False
    num_layers: int = 26
    hidden_size: int = 1152
    num_attention_heads: int = 4
    num_query_groups: int = 1
    kv_channels: int = 256
    ffn_hidden_size: int = 6912
    window_size: int = 512
    rope_scaling_factor: float = 1.0  # no rope scaling
    seq_length: int = 32768
    bf16: bool = True
    vocab_size: int = 262_144


@dataclass
class Gemma3ModelProvider4B(GemmaModelProvider):
    """Gemma3 4B config"""

    is_vision_language: bool = True
    num_layers: int = 34
    hidden_size: int = 2560
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    ffn_hidden_size: int = 10240
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208

@dataclass
class Gemma3ModelProvider12B(GemmaModelProvider):
    """Gemma3 12B config"""

    is_vision_language: bool = True
    num_layers: int = 48
    hidden_size: int = 3840
    num_attention_heads: int = 16
    num_query_groups: int = 8
    kv_channels: int = 256
    ffn_hidden_size: int = 15360
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208


@dataclass
class Gemma3ModelProvider27B(GemmaModelProvider):
    """Gemma3 27B config"""

    is_vision_language: bool = True
    num_layers: int = 62
    hidden_size: int = 5376
    num_attention_heads: int = 32
    num_query_groups: int = 16
    kv_channels: int = 128
    softmax_scale: int = 1.0 / math.sqrt(168)  # only for 27B, (5376 // 32)^(-0.5)
    ffn_hidden_size: int = 21504
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208