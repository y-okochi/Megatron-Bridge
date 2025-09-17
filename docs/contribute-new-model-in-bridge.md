# Contribute a New Model to Megatron Bridge

This guide explains how to add support for a new 🤗 Hugging Face model (or family) to Megatron Bridge so it can convert HF ↔ Megatron-Core formats and participate in training recipes.

Use this checklist-style flow: scaffold → provider mapping → parameter mappings → tests → validation.


## Prerequisites

- Familiarity with the Megatron Bridge repository structure.
- A working Python 3.10+ environment with Megatron Bridge installed (see [installation instructions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/README.md#installation)), a container is recommended.
- Familiarity with Megatron-Core GPT-style modules and 🤗 Transformers config objects.
- Access to a small HF checkpoint for local testing.
- Read first:
  - [Bridge user guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
  - [Technical details](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
  - [Model bridges overview](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/README.md)


## 1) Decide the integration strategy

Most GPT-style models (like Qwen family and Llama family) can reuse Megatron-Core GPT Model by just mapping configuration. If the model requires custom building blocks (e.g., attention variant, RoPE variant, VLM modules), add a lightweight specialization similar to how 🤗 implements `modeling_xxx.py`.

- **Standard GPT-style models**: Implement a `Provider` and a `Bridge`. For example, see the [Llama provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_provider.py) and [Llama bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py).
- **Models with custom components**: If your model has custom operations or blocks (e.g., a unique attention mechanism), add a minimal modeling module in the same directory and reference it from the `Provider`. (example to be provided)


## 2) Scaffold the model folder

Create a folder under `src/megatron/bridge/models/<your_model>/` and add:

- `<your_model>_provider.py`: builds a `TransformerConfig`-compatible provider (or a subclass of an existing provider) and exposes `.provide_distributed_model()`. For example: [Llama provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_provider.py), [Qwen provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen_provider.py) or [Qwen2 provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_provider.py).
- `<your_model>_bridge.py`: architecture-specific bridge that maps HF config → provider and defines parameter mappings. For example: [Llama bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py), [Qwen3 bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen3_bridge.py) or [Qwen2 bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_bridge.py).
- Optional: `README.md` with any model quirks. For example: [Llama README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/README.md).

## 3) Implement the Provider

Your provider maps the Hugging Face config to Megatron-Core transformer config fields and lazily constructs the distributed model(s). Start from the generic GPT provider (`src/megatron/bridge/models/gpt_provider.py`) and specialize the necessary fields and flags:

- Parallelism: `tensor_model_parallel_size`, `pipeline_model_parallel_size`, optional VPP/EP settings.
- Numerics: `fp16`, `bf16`, `params_dtype`, activation recomputation.
- Architecture quirks: RoPE base/scale, QK layernorm, tied embeddings, KV groups, max sequence length, etc.
- Optional custom modules: point to custom attention/MLP implementations using a layer spec if needed.

Expose:
```python
provider = YourModelProvider(...)
model = provider.provide_distributed_model(wrap_with_ddp=False)
```


## 4) Define Config and Parameter Mappings

Use the `provider_bridge` method to map Hugging Face configs to a Megatron model provider, and use `MegatronMappingRegistry` to map Megatron parameter names to Hugging Face parameter names. Start with the essentials (embeddings, final norm, QKV, MLP), then add extras (biases, rotary embeddings, experts, and vision blocks).

- `provider_bridge`: see [model_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/model_bridge.py)
- `MegatronMappingRegistry`: see [mapping_registry.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/mapping_registry.py)
- Mapping implementations: see [param_mapping.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/param_mapping.py)
- Background: see [Bridge technical details](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)

Example registration skeleton:

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM  # replace with your HF class
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from .<your_model>_provider import YourModelProvider

@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class YourModelBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> YourModelProvider:
        cfg = hf_pretrained.config
        return YourModelProvider(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=getattr(cfg, "intermediate_size", 4 * cfg.hidden_size),
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            # set dtype flags via helper if needed
            params_dtype=self.dtype_from_hf(cfg),
            ...
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            ...
        )
```

Notes:
- Use `*` wildcards for per-layer patterns; the number of wildcards must match between `megatron_param` and the HF pattern(s).
- `*` typically captures layer indices; `**` can match across dots. For example, to map both `.weight` and `.bias` together:
  ```python
  AutoMapping(
      megatron_param="output_layer.**",
      hf_param="lm_head.**",
  ),
  ```
- Prefer `AutoMapping` when the Megatron layer type implies the TP split automatically.
- Use `QKVMapping` for fused QKV and `GatedMLPMapping` for gate/up concatenation.

## 5) Minimal smoke test (local)

A minimal bidirectional end-to-end check:
```python
from megatron.bridge import AutoBridge

# HF → Megatron
bridge = AutoBridge.from_hf_pretrained("<org>/<model-id>", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)

# Megatron → HF (stream a few tensors)
for i, (name, tensor) in enumerate(bridge.export_hf_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10:
        break
```


## 6) Validate with examples (optional)
Use the examples in `examples/models/` to verify bidirectional conversion and basic generation with more complex model parallel setups.


- Generate from HF directly with the bridge
- Convert checkpoints back and forth
- Multi-GPU HF load to Megatron

```sh
python examples/models/generate_from_hf.py --model <org>/<model-id> | cat
python examples/models/checkpoint_conversion.py --model <org>/<model-id> | cat
```
## 7) Add tests

Add or extend tests under `tests/functional_tests/models/` and `tests/unit_tests/models/`:

- Conversion coverage:
  - HF → Megatron load succeeds without missing params
  - Megatron → HF export round-trips shapes and dtypes
- Provider coverage:
  - Provider fields align with HF config (heads, groups, ffn size, rope)
- Optional numeric checks (N):
  - Forward parity on a handful of tokens comparing HF vs Megatron outputs

Examples to reference:
- `tests/functional_tests/models/test_qwen3_provider.py`
- `tests/functional_tests/models/test_qwen3_conversion.py`

Run fast tests locally:
```sh
uv run pytest -q tests/functional_tests/models/test_<your_model>_provider.py -k your_model | cat
uv run pytest -q tests/functional_tests/models/test_<your_model>_conversion.py -k your_model | cat
```

Full suite (slower):
```sh
uv run pytest -q tests | cat
```


## 8) Troubleshooting

- Shape mismatches: double-check TP/PP splits and `QKVMapping` head ordering.
- Missing weights: ensure every Megatron param has a mapping; print unresolved names.
- Dtype issues: cast HF weights to destination dtype inside mappings when needed.
- EP/MoE layers: see EP-specific gather/scatter helpers in `param_mapping.py`.
- RoPE variants: normalize rope base/scale in the provider to match Megatron expectations.

Enable verbose logs:
```python
import logging
logging.getLogger("megatron.bridge").setLevel(logging.DEBUG)
```


## 9) PR checklist

- Provider maps all required config fields
- All critical parameters are covered by mappings
- Functional tests added and green
- Docs updated: this page and, if needed, a short `README.md` in your model folder
- Add your model to the Supported Models table in the repo `README.md` if applicable


## 10) Useful links

- User guide: [docs/bridge-guide.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
- Technical deep-dive: [docs/bridge-tech-details.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
- Code examples: [examples/models/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models)
- Providers and bridges: [src/megatron/bridge/models/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models)
- GitHub source tree: [Megatron Bridge src/megatron/bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge)

