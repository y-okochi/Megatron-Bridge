import torch
from transformers import AutoTokenizer
from rich.progress import track

from megatron.hub import CausalLMBridge

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
bridge = CausalLMBridge.from_pretrained(HF_MODEL_ID)


def generate_sequence(prompt, model, hf_model_path, max_new_tokens=100):
    """Generate text sequence"""
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.cuda()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
        0
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)

    generated_tokens = []
    cur_input_ids = input_ids
    cur_position_ids = position_ids
    cur_attention_mask = attention_mask

    for _ in track(range(max_new_tokens), description="Generating..."):
        # Move inputs to GPU
        cur_input_ids = cur_input_ids.cuda()
        cur_position_ids = cur_position_ids.cuda()
        cur_attention_mask = cur_attention_mask.cuda()

        # Forward inference with the model
        with torch.no_grad():
            model[0].cuda()
            output = model[0].module(
                cur_input_ids, cur_position_ids, cur_attention_mask
            )

        # Get the next token
        next_token = output.argmax(dim=-1)[:, -1]
        generated_tokens.append(next_token.item())

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Update input sequence
        cur_input_ids = torch.cat([cur_input_ids, next_token.unsqueeze(0)], dim=1)
        cur_position_ids = torch.arange(
            cur_input_ids.shape[1], device=cur_input_ids.device
        ).unsqueeze(0)
        cur_attention_mask = torch.ones_like(cur_input_ids, dtype=torch.bool)

    # Decode the generated token sequence
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text:\n{generated_text}")
    return generated_text


if __name__ == "__main__":
    model_provider = bridge.to_megatron()
    model = model_provider(wrap_with_ddp=False)

    prompt = "Hello, how are you?"
    generate_sequence(prompt, model, HF_MODEL_ID)