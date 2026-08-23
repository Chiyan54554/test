"""Encode UTF-8 text into a compact uint16 token stream for language-model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text with a SentencePiece tokenizer.")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model file")
    parser.add_argument("--input", nargs="+", required=True, help="UTF-8 text files")
    parser.add_argument("--output", required=True, help="output .bin file")
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if tokenizer.vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("this compact format supports vocabularies up to 65,535 tokens")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_written = 0
    with output_path.open("wb") as output_file:
        for input_name in args.input:
            with Path(input_name).open("r", encoding="utf-8") as input_file:
                for line in input_file:
                    token_ids = tokenizer.encode(line.strip(), out_type=int)
                    if token_ids:
                        token_ids.append(tokenizer.eos_id())
                        np.asarray(token_ids, dtype=np.uint16).tofile(output_file)
                        tokens_written += len(token_ids)

    print(f"wrote {tokens_written:,} tokens to {output_path}")


if __name__ == "__main__":
    main()
