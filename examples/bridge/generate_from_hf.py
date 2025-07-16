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

"""
Example:
  python examples/bridge/generate_from_hf.py --hf_model_path="/path/to/hf_model" --prompt="Hello, how are you?"
"""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from nemo.utils.get_rank import get_last_rank
from transformers import AutoTokenizer

from megatron.hub import CausalLMBridge


class SingleBatchIterator:
    def __init__(self, input_ids, position_ids, attention_mask):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def main(args) -> None:
    # pylint: disable=C0115,C0116
    tp = args.tp
    pp = args.pp

    # Load model from HuggingFace
    bridge = CausalLMBridge.from_hf_pretrained(args.hf_model_path)
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.initialize_model_parallel(seed=0)
    model = model_provider(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the input prompt
    prompt = args.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            if torch.distributed.get_rank() == 0:
                print(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids, attention_mask)

            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug: print token information
                if step < 5:  # Only for first few iterations
                    print(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print(f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})")
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(list(generated_ids[0]))
    if torch.distributed.get_rank() == 0:
        print("======== GENERATED TEXT OUTPUT ========")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation from HuggingFace Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="Llama-3.2-1B",
        help="Path to the HuggingFace model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallelism size')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline parallelism size')
    args = parser.parse_args()

    main(args)
