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
    parser.add_argument("--progress-every", type=int, default=1000, help="print progress every N lines")
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be a positive integer")

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    if tokenizer.vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("this compact format supports vocabularies up to 65,535 tokens")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    tokens_written = 0
    lines_processed = 0
    with temporary_path.open("wb") as output_file:
        for input_name in args.input:
            with Path(input_name).open("r", encoding="utf-8") as input_file:
                for line in input_file:
                    lines_processed += 1
                    token_ids = tokenizer.encode(line.strip(), out_type=int)
                    if token_ids:
                        token_ids.append(tokenizer.eos_id())
                        np.asarray(token_ids, dtype=np.uint16).tofile(output_file)
                        tokens_written += len(token_ids)
                    if lines_processed % args.progress_every == 0:
                        print(f"encoded {lines_processed:,} lines ({tokens_written:,} tokens)", flush=True)

    temporary_path.replace(output_path)
    print(f"wrote {tokens_written:,} tokens to {output_path}")


if __name__ == "__main__":
    main()
