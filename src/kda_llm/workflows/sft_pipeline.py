"""One-command SFT data preparation and fine-tuning workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_module, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, encode, and fine-tune a KDA checkpoint in one command.")
    parser.add_argument("--checkpoint", required=True, help="pretraining checkpoint to fine-tune")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece tokenizer used for pretraining")
    parser.add_argument("--sources", default="configs/sft_sources.json")
    parser.add_argument("--work-dir", default="runs/sft")
    parser.add_argument("--limit", type=int, default=40_000, help="maximum downloaded SFT examples")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1, help="number of micro-batches per optimizer update")
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_length <= 0 or args.progress_every <= 0:
        parser.error("limit, max length, and progress interval must be positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.lr <= 0 or args.log_every <= 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("invalid SFT hyperparameters")

    work_dir = Path(args.work_dir)
    state_dir = work_dir / ".pipeline_state"
    raw_path, dataset_path = work_dir / "raw.jsonl", work_dir / "sft.pt"
    checkpoint_dir = work_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / f"kda-sft-epoch-{args.epochs}.pt"
    state_dir.mkdir(parents=True, exist_ok=True)

    run_stage(
        "download_sft",
        state_dir,
        (raw_path,),
        args.resume,
        lambda: run_module(
            "kda_llm.cli.download_sft",
            "--sources", args.sources,
            "--output", str(raw_path),
            "--limit", str(args.limit),
            "--progress-every", str(args.progress_every),
        ),
    )
    run_stage(
        "prepare_sft",
        state_dir,
        (dataset_path,),
        args.resume,
        lambda: run_module(
            "kda_llm.cli.prepare_sft",
            "--input", str(raw_path),
            "--tokenizer", args.tokenizer,
            "--output", str(dataset_path),
            "--max-length", str(args.max_length),
        ),
    )
    sft_args = [
        "--checkpoint", args.checkpoint,
        "--dataset", str(dataset_path),
        "--out-dir", str(checkpoint_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--warmup-ratio", str(args.warmup_ratio),
        "--log-every", str(args.log_every),
        "--device", args.device,
        "--compile" if args.compile else "--no-compile",
        "--fused-cross-entropy" if args.fused_cross_entropy else "--no-fused-cross-entropy",
        "--fused-optimizer" if args.fused_optimizer else "--no-fused-optimizer",
    ]
    run_stage("sft", state_dir, (final_checkpoint,), args.resume, lambda: run_module("kda_llm.cli.sft", *sft_args))
    print(f"\nSFT complete: {final_checkpoint}")


if __name__ == "__main__":
    main()
