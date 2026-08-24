"""Command-line entry point for cache-aware KDA text generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.inference import GenerationConfig, generate, load_model, sample_next_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Traditional Chinese text from a KDA checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling options")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.checkpoint, device)
    if tokenizer.vocab_size() != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the checkpoint")
    completion = generate(model, tokenizer, args.prompt, GenerationConfig(args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed), device)
    print(completion)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.prompt + completion + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
