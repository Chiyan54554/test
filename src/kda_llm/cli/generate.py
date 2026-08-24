"""Generate text from a trained Chinese KDA checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.model import KDAConfig, KDALanguageModel


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Sample one token after temperature, top-k, and nucleus filtering."""
    logits = logits / temperature
    if top_k:
        cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probabilities - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Traditional Chinese text from a KDA checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="checkpoint written by kda-train")
    parser.add_argument("--tokenizer", required=True, help="matching SentencePiece .model file")
    parser.add_argument("--prompt", required=True, help="text used to begin generation")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50, help="0 disables top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", help="optional UTF-8 file for the completed text")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling options")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or "model" not in checkpoint:
        raise ValueError("checkpoint is not a kda-train checkpoint")
    config = KDAConfig(**checkpoint["config"])
    if tokenizer.vocab_size() != config.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match the checkpoint")

    model = KDALanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    prompt_ids = tokenizer.encode(args.prompt, out_type=int)
    if not prompt_ids:
        parser.error("--prompt must contain at least one token")
    prompt_ids = prompt_ids[-config.max_seq_len :]
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode():
        with autocast_context():
            logits, _, cache = model(token_ids, use_cache=True)
        position = token_ids.size(1)
        for token_index in range(args.max_new_tokens):
            next_token = sample_next_token(logits[:, -1], args.temperature, args.top_k, args.top_p)
            token_id = next_token.item()
            if token_id == tokenizer.eos_id():
                break
            generated_ids.append(token_id)
            if token_index + 1 < args.max_new_tokens:
                with autocast_context():
                    logits, _, cache = model(
                        next_token, past_states=cache, use_cache=True, position_offset=position
                    )
                position += 1

    completion = tokenizer.decode(generated_ids)
    print(completion)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(args.prompt + completion + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
