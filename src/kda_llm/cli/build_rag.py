"""Build a local BM25 index from Markdown and text reference files."""

from __future__ import annotations

import argparse

from kda_llm.retrieval import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local RAG index from .md and .txt files.")
    parser.add_argument("--input", required=True, help="reference file or directory")
    parser.add_argument("--output", required=True, help="output JSON index")
    parser.add_argument("--chunk-chars", type=int, default=600)
    parser.add_argument("--overlap-chars", type=int, default=80)
    args = parser.parse_args()
    try:
        files, chunks = build_index(args.input, args.output, args.chunk_chars, args.overlap_chars)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"indexed {chunks:,} chunks from {files:,} reference files into {args.output}")


if __name__ == "__main__":
    main()
