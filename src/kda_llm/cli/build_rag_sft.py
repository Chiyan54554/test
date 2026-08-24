"""Create answer-grounded SFT conversations from a local RAG index."""

from __future__ import annotations

import argparse

from kda_llm.data.rag_sft import build_rag_sft_records, write_rag_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context-grounded SFT JSONL from a local RAG index.")
    parser.add_argument("--index", required=True, help="JSON index from kda-build-rag")
    parser.add_argument("--output", required=True, help="output SFT JSONL")
    parser.add_argument("--examples-per-chunk", type=int, default=6)
    parser.add_argument("--context-chars", type=int, default=180)
    parser.add_argument("--answer-chars", type=int, default=180)
    args = parser.parse_args()
    try:
        records = build_rag_sft_records(args.index, args.examples_per_chunk, args.context_chars, args.answer_chars)
        write_rag_sft_jsonl(records, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {len(records):,} grounded SFT examples to {args.output}")


if __name__ == "__main__":
    main()
