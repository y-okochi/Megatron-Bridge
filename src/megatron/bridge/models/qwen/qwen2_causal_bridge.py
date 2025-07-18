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

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Qwen2ForCausalLM

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen2_provider import Qwen2ModelProvider
from megatron.bridge.models.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.model_bridge import MegatronModelBridge
from megatron.bridge.models.param_mapping import (
    GatedMLPMapping,
    QKVMapping,
    TPAwareMapping,
)


@MegatronModelBridge.register_bridge(source=Qwen2ForCausalLM, target=GPTModel)
class Qwen2CausalBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Qwen2 Causal LM.

    This bridge handles the conversion between HuggingFace Qwen2ForCausalLM
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2-7B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen2ModelProvider:
        hf_config = hf_pretrained.config

        provider = Qwen2ModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            add_qkv_bias=True,  # Qwen2 has bias in QKV projections
        )

        provider.gradient_accumulation_fusion = False
        provider.variable_seq_lengths = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            # ------------------------------------------------------------------
            # Embedding & output projection – column-parallel
            # ------------------------------------------------------------------
            TPAwareMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            TPAwareMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            # ------------------------------------------------------------------
            # LayerNorm (replicated across TP ranks)
            # ------------------------------------------------------------------
            TPAwareMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
            # ------------------------------------------------------------------
            # Attention – fused QKV & output projection
            # Note: Qwen2 has bias in QKV projections
            # ------------------------------------------------------------------
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                q="model.layers.*.self_attn.q_proj.bias",
                k="model.layers.*.self_attn.k_proj.bias",
                v="model.layers.*.self_attn.v_proj.bias",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            # ------------------------------------------------------------------
            # MLP – gated projection & output projection
            # ------------------------------------------------------------------
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
        )