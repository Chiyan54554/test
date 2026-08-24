"""Fail-fast diagnostics for the Linux CUDA training environment."""

from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command in a Linux NVIDIA GPU environment.")
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as error:
        raise SystemExit("chunk_kda is unavailable. Run `uv sync --extra cuda`.") from error
    print(f"torch: {torch.__version__}")
    print(f"cuda runtime: {torch.version.cuda}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"kda backend: {chunk_kda.__module__}.{chunk_kda.__name__}")


if __name__ == "__main__":
    main()
