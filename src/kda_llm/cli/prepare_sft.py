"""Encode JSONL conversations into fixed-length answer-masked SFT tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.data.sft import encode_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare answer-only masked tensors for SFT.")
    parser.add_argument("--input", required=True, help="SFT JSONL conversations")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece tokenizer")
    parser.add_argument("--output", required=True, help="output .pt tensor dataset")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    input_ids, labels, skipped = encode_jsonl(args.input, tokenizer, args.max_length)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "input_ids": input_ids, "labels": labels, "max_length": args.max_length}, path)
    print(f"wrote {input_ids.size(0):,} examples ({skipped:,} skipped) to {path}")


if __name__ == "__main__":
    main()
