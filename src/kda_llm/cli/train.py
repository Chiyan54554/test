"""Command-line entry point for the KDA training runtime."""

from __future__ import annotations

import argparse
import sys

from kda_llm.config import load_json_object
from kda_llm.models import KDAConfig
from kda_llm.training.engine import run_training


TRAIN_CONFIG_KEYS = {
    "max_tokens", "batch_size", "grad_accum", "seq_len", "lr", "warmup_steps", "save_every", "eval_every",
    "eval_steps", "seed", "log_every", "compile", "fused_cross_entropy", "fused_optimizer", "require_kda_kernel",
    "profile_start_step", "profile_warmup_steps", "profile_steps", "profile_dir",
}


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--train-config")
    bootstrap_args, _ = bootstrap.parse_known_args()
    parser = argparse.ArgumentParser(description="Train a 32M Chinese KDA language model.")
    parser.add_argument("--train-config", help="JSON file containing training hyperparameters")
    parser.add_argument("--train-data", default="runs/smoke/corpus/train.bin", help="single uint16 .bin token stream")
    parser.add_argument("--train-sources", help="JSON array of weighted uint16 token streams")
    parser.add_argument("--model-config", default="configs/model_32m.json", help="KDA architecture JSON")
    parser.add_argument("--val-data", default="runs/smoke/corpus/valid.bin", help="optional uint16 .bin validation stream")
    parser.add_argument("--out-dir", default="runs/smoke/checkpoints")
    parser.add_argument("--resume-from", help="checkpoint written by kda-train to continue from")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--steps", type=int, help="legacy explicit step count, intended only for smoke tests")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-cross-entropy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-kda-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-start-step", type=int)
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-dir", default="runs/smoke/profiles")
    if bootstrap_args.train_config:
        parser.set_defaults(**load_json_object(bootstrap_args.train_config, TRAIN_CONFIG_KEYS))
    args = parser.parse_args()
    args.train_data_was_set = "--train-data" in sys.argv[1:]
    args.model_config_values = load_json_object(args.model_config, set(KDAConfig.__dataclass_fields__))
    run_training(args, parser)


if __name__ == "__main__":
    main()
