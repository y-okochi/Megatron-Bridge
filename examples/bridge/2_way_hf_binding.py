"""
This example demonstrates how to use the CausalLMBridge to perform a round-trip
conversion between a Hugging Face model and a Megatron-LM model.

The process is as follows:
1. A CausalLMBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. The bridge's `to_megatron` method is called to get a Megatron-LM compatible model provider.
3. The model provider is used to instantiate the Megatron-LM model.
4. Finally, the `save_pretrained` method is used to save the Megatron-LM
    model back into the Hugging Face format. A new directory, named after the
    model, will be created for the converted model files. By default, this
    directory is created in the current working directory, but a different
    parent directory can be specified via the `--output-dir` argument.
"""
import argparse
import os

import torch
from rich.console import Console
from rich.table import Table

from megatron.hub import CausalLMBridge

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"


def main(hf_model_id: str = HF_MODEL_ID, output_dir: str = None) -> None:
    model_name = hf_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, model_name)
    else:
        save_path = model_name

    bridge = CausalLMBridge.from_pretrained(hf_model_id)

    # Formatting
    console = Console()
    table = Table(title="Hugging Face Weights Verification")
    table.add_column("Weight Name", style="cyan")
    table.add_column("Shape")
    table.add_column("DType")
    table.add_column("Device")
    table.add_column("Matches Original", justify="center")

    model_provider = bridge.to_provider()
    megatron_model = model_provider(wrap_with_ddp=False)

    # Check each weight against the original HF-model
    for name, param in bridge(megatron_model, show_progress=True):
        original_param = bridge.hf_pretrained.state[name]
        table.add_row(
            name,
            str(tuple(param.shape)),
            str(param.dtype).replace("torch.", ""),
            str(param.device),
            "✅"
            if torch.allclose(param, original_param.to(param.device), atol=1e-6)
            else "❌",
        )

    console.print(table)
    console.print(f"Saving HF-ckpt in {save_path}...")
    bridge.save_pretrained(megatron_model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert between HuggingFace and Megatron-LM model formats')
    parser.add_argument('--hf-model-id', type=str, default=HF_MODEL_ID,
                        help='HuggingFace model ID to convert')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='The directory where the converted model directory will be created. Defaults to the current working directory.')
    
    args = parser.parse_args()
    main(args.hf_model_id, args.output_dir)
