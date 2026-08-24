"""Command-line entry point for Traditional Chinese supervised fine-tuning."""

from __future__ import annotations

import argparse

from kda_llm.training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tune a KDA checkpoint on answer-masked conversations.")
    parser.add_argument("--checkpoint", required=True, help="base pretraining checkpoint")
    parser.add_argument("--dataset", required=True, help=".pt dataset from kda-prepare-sft")
    parser.add_argument("--out-dir", default="runs/sft/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1, help="number of micro-batches per optimizer update")
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0 or args.lr <= 0 or args.log_every <= 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("invalid SFT hyperparameters")
    run_sft(args)


if __name__ == "__main__":
    main()
