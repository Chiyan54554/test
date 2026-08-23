"""Run the complete Chinese KDA data-to-training smoke-test pipeline."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


def run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True)


def split_corpus(input_path: Path, train_path: Path, valid_path: Path, validation_ratio: float) -> None:
    train_count = 0
    valid_count = 0
    with input_path.open("r", encoding="utf-8") as input_file, train_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as train_file, valid_path.open("w", encoding="utf-8", newline="\n") as valid_file:
        for line in input_file:
            # Content hashing gives a stable split even when source order changes.
            bucket = int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")
            if bucket / 2**64 < validation_ratio:
                valid_file.write(line)
                valid_count += 1
            else:
                train_file.write(line)
                train_count += 1
    if not train_count or not valid_count:
        raise RuntimeError("split produced an empty train or validation corpus")
    print(f"split corpus: {train_count:,} train documents, {valid_count:,} validation documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Chinese KDA training pipeline.")
    parser.add_argument("--sources", default="configs/hf_sources.json", help="weighted Hugging Face source manifest")
    parser.add_argument("--total-documents", type=int, default=100_000, help="documents to stream before processing")
    parser.add_argument("--work-dir", default="runs/smoke", help="directory for all generated pipeline artifacts")
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.total_documents <= 1:
        parser.error("--total-documents must be greater than 1")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")

    work_dir = Path(args.work_dir)
    corpus_dir = work_dir / "corpus"
    tokenizer_dir = work_dir / "tokenizer"
    checkpoint_dir = work_dir / "checkpoints"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    raw_path = corpus_dir / "raw.txt"
    traditional_path = corpus_dir / "traditional.txt"
    train_text_path = corpus_dir / "train.txt"
    valid_text_path = corpus_dir / "valid.txt"
    tokenizer_prefix = tokenizer_dir / "chinese"
    train_bin_path = corpus_dir / "train.bin"
    valid_bin_path = corpus_dir / "valid.bin"

    run_module(
        "kda_llm.cli.download_hf_data",
        "--sources", args.sources,
        "--total-documents", str(args.total_documents),
        "--output", str(raw_path),
    )
    run_module(
        "kda_llm.cli.convert_traditional",
        "--input", str(raw_path),
        "--output", str(traditional_path),
    )
    split_corpus(traditional_path, train_text_path, valid_text_path, args.validation_ratio)
    run_module(
        "kda_llm.cli.build_tokenizer",
        "--input", str(train_text_path),
        "--output", str(tokenizer_prefix),
    )
    run_module(
        "kda_llm.cli.prepare_data",
        "--tokenizer", str(tokenizer_prefix) + ".model",
        "--input", str(train_text_path),
        "--output", str(train_bin_path),
    )
    run_module(
        "kda_llm.cli.prepare_data",
        "--tokenizer", str(tokenizer_prefix) + ".model",
        "--input", str(valid_text_path),
        "--output", str(valid_bin_path),
    )
    run_module(
        "kda_llm.cli.train",
        "--train-data", str(train_bin_path),
        "--val-data", str(valid_bin_path),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--device", args.device,
        "--out-dir", str(checkpoint_dir),
    )


if __name__ == "__main__":
    main()
