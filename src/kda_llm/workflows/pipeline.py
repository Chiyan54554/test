"""End-to-end corpus preparation and training workflow."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from kda_llm.config import load_json_object
from kda_llm.models import KDAConfig


TRAIN_CONFIG_KEYS = {
    "max_tokens", "batch_size", "grad_accum", "seq_len", "lr", "warmup_steps", "save_every", "eval_every",
    "eval_steps", "seed", "log_every", "compile", "fused_cross_entropy", "fused_optimizer", "require_kda_kernel",
    "profile_start_step", "profile_warmup_steps", "profile_steps", "profile_dir",
}


def run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True)


def run_stage(name: str, state_dir: Path, artifacts: tuple[Path, ...], resume: bool, action: Callable[[], None]) -> None:
    marker = state_dir / f"{name}.done"
    if resume and marker.is_file() and all(path.is_file() for path in artifacts):
        print(f"[resume] skipping completed stage: {name}")
        return
    marker.unlink(missing_ok=True)
    print(f"\n=== {name} ===")
    action()
    marker.write_text("completed\n", encoding="utf-8")


def split_corpus(input_path: Path, train_path: Path, valid_path: Path, validation_ratio: float) -> None:
    train_count = valid_count = 0
    train_partial, valid_partial = train_path.with_suffix(train_path.suffix + ".partial"), valid_path.with_suffix(valid_path.suffix + ".partial")
    with input_path.open("r", encoding="utf-8") as source, train_partial.open("w", encoding="utf-8", newline="\n") as train, valid_partial.open("w", encoding="utf-8", newline="\n") as valid:
        for line in source:
            bucket = int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")
            if bucket / 2**64 < validation_ratio:
                valid.write(line)
                valid_count += 1
            else:
                train.write(line)
                train_count += 1
    if not train_count or not valid_count:
        raise RuntimeError("split produced an empty train or validation corpus")
    train_partial.replace(train_path)
    valid_partial.replace(valid_path)
    print(f"split corpus: {train_count:,} train documents, {valid_count:,} validation documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Chinese KDA training pipeline.")
    parser.add_argument("--sources", default="configs/hf_sources.json")
    parser.add_argument("--model-config", default="configs/model_32m.json")
    parser.add_argument("--train-config", default="configs/train_gpu.json")
    parser.add_argument("--total-documents", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--work-dir", default="runs/smoke")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--clean-min-chars", type=int, default=20)
    parser.add_argument("--clean-max-chars", type=int, default=20_000)
    parser.add_argument("--clean-min-cjk-ratio", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-dir")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    settings = load_json_object(args.train_config, TRAIN_CONFIG_KEYS)
    for key, value in settings.items():
        if key == "max_tokens" and "--steps" in sys.argv[1:]:
            continue
        if f"--{key.replace('_', '-')}" not in sys.argv[1:] and f"--no-{key.replace('_', '-')}" not in sys.argv[1:]:
            setattr(args, key, value)
    if args.total_documents <= 1 or args.progress_every <= 0 or not 0 < args.validation_ratio < 1:
        parser.error("invalid document or validation settings")
    if args.max_tokens is not None and args.steps is not None:
        parser.error("use either --max-tokens or --steps, not both")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    model_config = KDAConfig(**load_json_object(args.model_config, set(KDAConfig.__dataclass_fields__)))
    work_dir = Path(args.work_dir)
    corpus_dir, tokenizer_dir, checkpoint_dir, state_dir = work_dir / "corpus", work_dir / "tokenizer", work_dir / "checkpoints", work_dir / ".pipeline_state"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    raw, traditional, clean = corpus_dir / "raw.txt", corpus_dir / "traditional.txt", corpus_dir / "clean.txt"
    train_text, valid_text = corpus_dir / "train.txt", corpus_dir / "valid.txt"
    tokenizer_prefix = tokenizer_dir / "chinese"
    train_bin, valid_bin = corpus_dir / "train.bin", corpus_dir / "valid.bin"
    run_stage("download", state_dir, (raw,), args.resume, lambda: run_module("kda_llm.cli.download_hf_data", "--sources", args.sources, "--total-documents", str(args.total_documents), "--progress-every", str(args.progress_every), "--output", str(raw)))
    run_stage("convert", state_dir, (traditional,), args.resume, lambda: run_module("kda_llm.cli.convert_traditional", "--input", str(raw), "--output", str(traditional), "--progress-every", str(args.progress_every)))
    stats = clean.with_suffix(clean.suffix + ".stats.json")
    run_stage("clean", state_dir, (clean, stats), args.resume, lambda: run_module("kda_llm.cli.clean_corpus", "--input", str(traditional), "--output", str(clean), "--min-chars", str(args.clean_min_chars), "--max-chars", str(args.clean_max_chars), "--min-cjk-ratio", str(args.clean_min_cjk_ratio), "--progress-every", str(args.progress_every)))
    run_stage("split", state_dir, (train_text, valid_text), args.resume, lambda: split_corpus(clean, train_text, valid_text, args.validation_ratio))
    run_stage("tokenizer", state_dir, (Path(str(tokenizer_prefix) + ".model"), Path(str(tokenizer_prefix) + ".vocab")), args.resume, lambda: run_module("kda_llm.cli.build_tokenizer", "--input", str(train_text), "--output", str(tokenizer_prefix), "--vocab-size", str(model_config.vocab_size)))
    run_stage("encode_train", state_dir, (train_bin,), args.resume, lambda: run_module("kda_llm.cli.prepare_data", "--tokenizer", str(tokenizer_prefix) + ".model", "--input", str(train_text), "--output", str(train_bin), "--progress-every", str(args.progress_every)))
    run_stage("encode_valid", state_dir, (valid_bin,), args.resume, lambda: run_module("kda_llm.cli.prepare_data", "--tokenizer", str(tokenizer_prefix) + ".model", "--input", str(valid_text), "--output", str(valid_bin), "--progress-every", str(args.progress_every)))
    print("\n=== train ===")
    budget = ("--max-tokens", str(args.max_tokens)) if args.max_tokens is not None else ("--steps", str(args.steps))
    profile = ("--profile-start-step", str(args.profile_start_step), "--profile-warmup-steps", str(args.profile_warmup_steps), "--profile-steps", str(args.profile_steps), "--profile-dir", args.profile_dir or str(work_dir / "profiles")) if args.profile_steps else ()
    train_args = ["--train-data", str(train_bin), "--val-data", str(valid_bin), *budget, "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum), "--seq-len", str(args.seq_len), "--lr", str(args.lr), "--model-config", args.model_config, "--train-config", args.train_config, "--device", args.device, "--out-dir", str(checkpoint_dir)]
    if args.compile:
        train_args.append("--compile")
    if args.fused_cross_entropy:
        train_args.append("--fused-cross-entropy")
    train_args.append("--fused-optimizer" if args.fused_optimizer else "--no-fused-optimizer")
    if args.resume_checkpoint:
        train_args.extend(("--resume-from", args.resume_checkpoint))
    train_args.extend(profile)
    run_module("kda_llm.cli.train", *train_args)
