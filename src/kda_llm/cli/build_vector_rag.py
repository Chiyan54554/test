"""Build a cached multilingual embedding index from a local RAG index."""

from __future__ import annotations

import argparse

import torch

from kda_llm.retrieval.semantic import DEFAULT_EMBEDDING_MODEL, build_vector_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multilingual embedding index for hybrid RAG.")
    parser.add_argument("--index", required=True, help="JSON index created by kda-build-rag")
    parser.add_argument("--output", required=True, help="output .npz vector index")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    try:
        count = build_vector_index(args.index, args.output, args.embedding_model, device)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {count:,} vectors to {args.output}")


if __name__ == "__main__":
    main()
