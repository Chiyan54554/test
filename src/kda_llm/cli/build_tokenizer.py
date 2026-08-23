"""Train a Chinese-first SentencePiece BPE tokenizer from UTF-8 text files."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact Chinese BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="UTF-8 text corpus files")
    parser.add_argument("--output", default="tokenizer/chinese", help="output filename prefix")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"corpus files not found: {', '.join(missing)}")

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=','.join(str(path) for path in input_paths),
        model_prefix=str(output_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc_cf",
        split_digits=True,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


if __name__ == "__main__":
    main()
