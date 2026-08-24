"""One-command context-grounded SFT workflow for a local RAG index."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_module, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-SFT data and fine-tune a KDA checkpoint in one command.")
    parser.add_argument("--checkpoint", required=True, help="completed general-SFT checkpoint")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece tokenizer used by the checkpoint")
    parser.add_argument("--rag-index", required=True, help="JSON index from kda-build-rag")
    parser.add_argument("--work-dir", default="runs/rag_sft")
    parser.add_argument("--examples-per-chunk", type=int, default=6)
    parser.add_argument("--context-chars", type=int, default=180)
    parser.add_argument("--answer-chars", type=int, default=180)
    parser.add_argument("--refusal-ratio", type=float, default=0.25)
    parser.add_argument("--context-chunks", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.examples_per_chunk, args.context_chars, args.answer_chars, args.context_chunks, args.max_length, args.epochs, args.batch_size, args.log_every) <= 0:
        parser.error("RAG-SFT sizes must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1 or not 0 <= args.refusal_ratio < 1:
        parser.error("invalid RAG-SFT hyperparameters")

    work_dir = Path(args.work_dir)
    state_dir = work_dir / ".pipeline_state"
    raw_path, dataset_path = work_dir / "raw.jsonl", work_dir / "rag_sft.pt"
    checkpoint_dir = work_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / f"kda-sft-epoch-{args.epochs}.pt"
    state_dir.mkdir(parents=True, exist_ok=True)
    run_stage("build_rag_sft", state_dir, (raw_path,), args.resume, lambda: run_module("kda_llm.cli.build_rag_sft", "--index", args.rag_index, "--output", str(raw_path), "--examples-per-chunk", str(args.examples_per_chunk), "--context-chars", str(args.context_chars), "--answer-chars", str(args.answer_chars), "--refusal-ratio", str(args.refusal_ratio), "--context-chunks", str(args.context_chunks)))
    run_stage("prepare_rag_sft", state_dir, (dataset_path,), args.resume, lambda: run_module("kda_llm.cli.prepare_sft", "--input", str(raw_path), "--tokenizer", args.tokenizer, "--output", str(dataset_path), "--max-length", str(args.max_length)))
    sft_args = ["--checkpoint", args.checkpoint, "--dataset", str(dataset_path), "--out-dir", str(checkpoint_dir), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--weight-decay", str(args.weight_decay), "--warmup-ratio", str(args.warmup_ratio), "--log-every", str(args.log_every), "--device", args.device, "--compile" if args.compile else "--no-compile", "--fused-cross-entropy" if args.fused_cross_entropy else "--no-fused-cross-entropy", "--fused-optimizer" if args.fused_optimizer else "--no-fused-optimizer"]
    run_stage("rag_sft", state_dir, (final_checkpoint,), args.resume, lambda: run_module("kda_llm.cli.sft", *sft_args))
    print(f"\nRAG-SFT complete: {final_checkpoint}")


if __name__ == "__main__":
    main()
