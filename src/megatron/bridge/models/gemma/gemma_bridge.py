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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    GatedMLPMapping,
)
import torch
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.gemma.gemma_provider import GemmaModelProvider
from transformers import Gemma3ForCausalLM
from megatron.core.models.gpt.gpt_model import GPTModel
import math

@MegatronModelBridge.register_bridge(source=Gemma3ForCausalLM, target=GPTModel)
class GemmaModelBridge(MegatronModelBridge):

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GemmaModelProvider:
        hf_config = hf_pretrained.config

        provider = GemmaModelProvider(
            init_method_std=hf_config.initializer_range,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            kv_channels=hf_config.head_dim,
            seq_length=hf_config.max_position_embeddings,
            num_attention_heads=hf_config.num_attention_heads,
            num_layers=hf_config.num_hidden_layers,
            num_query_groups=hf_config.num_key_value_heads,
            window_size=hf_config.sliding_window,
            rotary_base=(hf_config.rope_local_base_freq, hf_config.rope_theta),
            rms_norm_eps=hf_config.rms_norm_eps,
            vocab_size=hf_config.vocab_size,
            softmax_scale=1.0 / math.sqrt(hf_config.query_pre_attn_scalar),
            rope_scaling_factor=hf_config.rope_scaling.factor,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16), # TODO confirm 
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
        )

        return provider
    
    def mapping_registry(self) -> MegatronMappingRegistry:
        
        mapping_list = [
                # word emebdding
                AutoMapping("model.embed_tokens.weight", "model.embed_tokens.weight"),      
                # attention
                AutoMapping("model.layers.*.input_layernorm.weight", "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight"),
                AutoMapping("model.layers.*.self_attn.q_norm.weight", "decoder.layers.*.self_attention.q_layernorm.weight"),
                AutoMapping("model.layers.*.self_attn.k_norm.weight", "decoder.layers.*.self_attention.k_layernorm.weight"),
                AutoMapping("model.layers.*.self_attn.o_proj.weight", "decoder.layers.*.self_attention.linear_proj.weight"),
                AutoMapping("model.layers.*.post_attention_layernorm.weight", "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"),
                # mlp
                AutoMapping("model.layers.*.pre_feedforward_layernorm.weight", "decoder.layers.*.mlp.linear_fc1.layer_norm_weight"),
                AutoMapping("model.layers.*.mlp.down_proj.weight", "decoder.layers.*.mlp.linear_fc2.weight"),
                AutoMapping("model.layers.*.post_feedforward_layernorm.weight", "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"),
                # final norm
                AutoMapping("model.norm.weight", "decoder.final_layernorm.weight"),
        ]

        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

