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

import json
from functools import cached_property, partial
from pathlib import Path
from typing import Tuple, Union, Dict, Optional

import torch
import torch.nn as nn
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GenerationConfig, GptOssForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
    QKVMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.gpt_oss.gpt_oss_provider import GPTOSSProvider, quick_gelu
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source=GptOssForCausalLM, target=GPTModel)
class GPTOSSBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for GPT-OSS models.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("openai/gpt-oss-model")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTOSSProvider:
        hf_config = hf_pretrained.config

        # Extract generation config
        generation_config = getattr(hf_pretrained, 'generation_config', None)
        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(str(hf_pretrained.name_or_path))
            except Exception:
                generation_config = None

        # Extract rope scaling parameters
        rope_scaling = getattr(hf_config, 'rope_scaling', {})

        provider = GPTOSSProvider(
            # Basic model architecture
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            ffn_hidden_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,

            # MoE configuration
            num_moe_experts=getattr(hf_config, 'num_local_experts', None),
            moe_router_topk=getattr(hf_config, 'num_experts_per_tok', 4),
            moe_ffn_hidden_size=hf_config.intermediate_size,
            moe_grouped_gemm=True,  # Default for GPT-OSS
            moe_router_load_balancing_type="none",  # Based on router_aux_loss_coef presence
            moe_token_dispatcher_type="alltoall",

            # Attention configuration
            attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
            hidden_dropout=getattr(hf_config, 'hidden_dropout', 0.0),

            # Position embedding and rope configuration
            position_embedding_type="yarn",
            rotary_base=getattr(hf_config, 'rope_theta', 150000),
            rotary_scaling_factor=rope_scaling.get('factor', 32.0),
            yarn_original_max_position_embeddings=rope_scaling.get('original_max_position_embeddings', 4096),
            yarn_beta_fast=rope_scaling.get('beta_fast', 32.0),
            yarn_beta_slow=rope_scaling.get('beta_slow', 1.0),

            # Sliding window attention
            window_size=(getattr(hf_config, 'sliding_window', 128), 0),
            window_attn_skip_freq=2,  # Based on layer_types alternating pattern

            # Normalization
            normalization="RMSNorm",
            layernorm_epsilon=getattr(hf_config, 'rms_norm_eps', 1e-5),

            # Activation and GLU configuration
            activation_func=quick_gelu,  # Based on "hidden_act": "silu" in HF config
            glu_linear_offset=getattr(hf_config, 'swiglu_limit', 7.0),
            gated_linear_unit=True,

            # Attention configuration (continued)
            softmax_type="learnable",  # GPT-OSS specific

            # Other configurations
            init_method_std=getattr(hf_config, 'initializer_range', 0.02),
            share_embeddings_and_output_weights=getattr(hf_config, 'tie_word_embeddings', False),

            # Precision and performance
            bf16=True,
            params_dtype=torch.bfloat16,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),

            # Generation config
            generation_config=generation_config,
        )


        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Based on the GPT-OSS importer code provided.
        """

        # Dictionary maps HF parameter names -> Megatron parameter names
        # Based on the mapping from the OpenAI GPT-OSS importer
        param_mappings = {
            "embedding.weight": "embedding.word_embeddings.weight",
            "norm.scale": "decoder.final_layernorm.weight",
            "unembedding.weight": "output_layer.weight",
            "model.layers.*.attn.norm.scale": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.attn.out.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            "model.layers.*.attn.out.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.attn.sinks": "decoder.layers.*.self_attention.core_attention.softmax_offset",
            "model.layers.*.mlp.norm.scale": "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.mlp.gate.bias": "decoder.layers.*.mlp.router.bias",
            "model.layers.*.mlp.gate.weight": "decoder.layers.*.mlp.router.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend([
            QKVMapping(
                q="model.layers.*.attn.q_proj.*",
                k="model.layers.*.attn.k_proj.*",
                v="model.layers.*.attn.v_proj.*",
                megatron_param="decoder.layers.*.self_attention.linear_qkv.*",
            ),
            GPTOSSMLPDownProjMapping(
                hf_param="model.layers.*.mlp.experts.down_proj",
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
            ),
            GPTOSSMLPDownProjMapping(
                hf_param="model.layers.*.mlp.experts.down_proj_bias",
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.bias*",
            ),
            GPTOSSMLPGateUpProjMapping(
                hf_param="model.layers.*.mlp.experts.gate_up_proj",
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            ),
            GPTOSSMLPGateUpProjMapping(
                hf_param="model.layers.*.mlp.experts.gate_up_proj_bias",
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.bias*",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)


class GPTOSSMLPDownProjMapping(AutoMapping):
    """
    MLPDownProj for expert weights GPT-OSS models.
    """
    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return super().hf_to_megatron(hf_weights, megatron_module)
    
    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return super().megatron_to_hf(megatron_weights, megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass

class GPTOSSMLPGateUpProjMapping(MegatronParamMapping):
    """
    MLPGateUpProj for expert weights GPT-OSS models.
    """
    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        breakpoint()
        return super().hf_to_megatron(hf_weights, megatron_module)
    
    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return super().megatron_to_hf(megatron_weights, megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass