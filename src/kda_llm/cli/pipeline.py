"""Run the complete Chinese KDA data-to-training smoke-test pipeline."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from kda_llm.config import load_json_object
from kda_llm.model import KDAConfig

def run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True)


def run_stage(
    name: str,
    state_dir: Path,
    artifacts: tuple[Path, ...],
    resume: bool,
    action: Callable[[], None],
) -> None:
    marker_path = state_dir / f"{name}.done"
    if resume and marker_path.is_file() and all(path.is_file() for path in artifacts):
        print(f"[resume] skipping completed stage: {name}")
        return
    marker_path.unlink(missing_ok=True)
    print(f"\n=== {name} ===")
    action()
    marker_path.write_text("completed\n", encoding="utf-8")


def validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    print(f"training device: {selected}")


def split_corpus(input_path: Path, train_path: Path, valid_path: Path, validation_ratio: float) -> None:
    train_count = 0
    valid_count = 0
    train_temporary_path = train_path.with_suffix(train_path.suffix + ".partial")
    valid_temporary_path = valid_path.with_suffix(valid_path.suffix + ".partial")
    with input_path.open("r", encoding="utf-8") as input_file, train_temporary_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as train_file, valid_temporary_path.open("w", encoding="utf-8", newline="\n") as valid_file:
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
    train_temporary_path.replace(train_path)
    valid_temporary_path.replace(valid_path)
    print(f"split corpus: {train_count:,} train documents, {valid_count:,} validation documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Chinese KDA training pipeline.")
    parser.add_argument("--sources", default="configs/hf_sources.json", help="weighted Hugging Face source manifest")
    parser.add_argument("--model-config", default="configs/model_32m.json", help="KDA architecture JSON")
    parser.add_argument("--train-config", default="configs/train_gpu.json", help="training hyperparameters JSON")
    parser.add_argument("--total-documents", type=int, default=100_000, help="documents to stream before processing")
    parser.add_argument("--progress-every", type=int, default=1000, help="download progress interval in documents")
    parser.add_argument("--work-dir", default="runs/smoke", help="directory for all generated pipeline artifacts")
    parser.add_argument("--resume-checkpoint", help="checkpoint path passed to kda-train")
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--clean-min-chars", type=int, default=20)
    parser.add_argument("--clean-max-chars", type=int, default=20_000)
    parser.add_argument("--clean-min-cjk-ratio", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, help="total training tokens; derives optimizer steps")
    parser.add_argument("--steps", type=int, help="legacy smoke-test step count")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-start-step", type=int, help="global optimizer step at which to start the GPU profiler")
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-dir", help="directory for profiler traces and summaries")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed data preparation stages",
    )
    args = parser.parse_args()
    training_config = load_json_object(
        args.train_config,
        {"max_tokens", "batch_size", "grad_accum", "seq_len", "lr", "warmup_steps", "save_every", "eval_every", "eval_steps", "seed", "log_every", "compile", "fused_cross_entropy", "require_kda_kernel", "profile_start_step", "profile_warmup_steps", "profile_steps", "profile_dir"},
    )
    for key, value in training_config.items():
        option = f"--{key.replace('_', '-')}"
        negative_option = f"--no-{key.replace('_', '-')}"
        if option not in sys.argv[1:] and negative_option not in sys.argv[1:]:
            setattr(args, key, value)
    model_config = KDAConfig(**load_json_object(args.model_config, set(KDAConfig.__dataclass_fields__)))

    if args.total_documents <= 1:
        parser.error("--total-documents must be greater than 1")
    if args.progress_every <= 0:
        parser.error("--progress-every must be a positive integer")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    validate_device(args.device)

    work_dir = Path(args.work_dir)
    corpus_dir = work_dir / "corpus"
    tokenizer_dir = work_dir / "tokenizer"
    checkpoint_dir = work_dir / "checkpoints"
    state_dir = work_dir / ".pipeline_state"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    raw_path = corpus_dir / "raw.txt"
    traditional_path = corpus_dir / "traditional.txt"
    clean_path = corpus_dir / "clean.txt"
    clean_stats_path = clean_path.with_suffix(clean_path.suffix + ".stats.json")
    train_text_path = corpus_dir / "train.txt"
    valid_text_path = corpus_dir / "valid.txt"
    tokenizer_prefix = tokenizer_dir / "chinese"
    train_bin_path = corpus_dir / "train.bin"
    valid_bin_path = corpus_dir / "valid.bin"

    run_stage(
        "download", state_dir, (raw_path,), args.resume,
        lambda: run_module(
            "kda_llm.cli.download_hf_data", "--sources", args.sources,
            "--total-documents", str(args.total_documents), "--progress-every", str(args.progress_every),
            "--output", str(raw_path),
        ),
    )
    run_stage(
        "convert", state_dir, (traditional_path,), args.resume,
        lambda: run_module(
            "kda_llm.cli.convert_traditional", "--input", str(raw_path), "--output", str(traditional_path),
            "--progress-every", str(args.progress_every),
        ),
    )
    run_stage(
        "clean", state_dir, (clean_path, clean_stats_path), args.resume,
        lambda: run_module(
            "kda_llm.cli.clean_corpus", "--input", str(traditional_path), "--output", str(clean_path),
            "--min-chars", str(args.clean_min_chars), "--max-chars", str(args.clean_max_chars),
            "--min-cjk-ratio", str(args.clean_min_cjk_ratio), "--progress-every", str(args.progress_every),
        ),
    )
    run_stage(
        "split", state_dir, (train_text_path, valid_text_path), args.resume,
        lambda: split_corpus(clean_path, train_text_path, valid_text_path, args.validation_ratio),
    )
    run_stage(
        "tokenizer", state_dir, (Path(str(tokenizer_prefix) + ".model"), Path(str(tokenizer_prefix) + ".vocab")), args.resume,
        lambda: run_module(
            "kda_llm.cli.build_tokenizer", "--input", str(train_text_path), "--output", str(tokenizer_prefix),
            "--vocab-size", str(model_config.vocab_size),
        ),
    )
    run_stage(
        "encode_train", state_dir, (train_bin_path,), args.resume,
        lambda: run_module(
            "kda_llm.cli.prepare_data", "--tokenizer", str(tokenizer_prefix) + ".model",
            "--input", str(train_text_path), "--output", str(train_bin_path),
            "--progress-every", str(args.progress_every),
        ),
    )
    run_stage(
        "encode_valid", state_dir, (valid_bin_path,), args.resume,
        lambda: run_module(
            "kda_llm.cli.prepare_data", "--tokenizer", str(tokenizer_prefix) + ".model",
            "--input", str(valid_text_path), "--output", str(valid_bin_path),
            "--progress-every", str(args.progress_every),
        ),
    )
    print("\n=== train ===")
    compile_arguments = ("--compile",) if args.compile else ()
    fused_loss_arguments = ("--fused-cross-entropy",) if args.fused_cross_entropy else ()
    resume_arguments = ("--resume-from", args.resume_checkpoint) if args.resume_checkpoint else ()
    profile_arguments = (
        "--profile-start-step", str(args.profile_start_step),
        "--profile-warmup-steps", str(args.profile_warmup_steps),
        "--profile-steps", str(args.profile_steps),
        "--profile-dir", args.profile_dir or str(work_dir / "profiles"),
    ) if args.profile_steps else ()
    budget_arguments = ("--max-tokens", str(args.max_tokens)) if args.max_tokens is not None else ("--steps", str(args.steps))
    run_module(
        "kda_llm.cli.train",
        "--train-data", str(train_bin_path),
        "--val-data", str(valid_bin_path),
        *budget_arguments,
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--seq-len", str(args.seq_len),
        "--lr", str(args.lr),
        "--model-config", args.model_config,
        "--train-config", args.train_config,
        "--device", args.device,
        "--out-dir", str(checkpoint_dir),
        *compile_arguments,
        *fused_loss_arguments,
        *resume_arguments,
        *profile_arguments,
    )


if __name__ == "__main__":
    main()
