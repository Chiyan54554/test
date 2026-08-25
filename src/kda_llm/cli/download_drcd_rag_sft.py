"""Stream DRCD grounded QA examples from Hugging Face."""

from __future__ import annotations

import argparse

from datasets import load_dataset

from kda_llm.data.grounded_sft import normalize_drcd_record
from kda_llm.data.sft import write_jsonl


DEFAULT_DATASET = "steven0226/drcd-zhtw-extractive-qa-sft"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Traditional Chinese DRCD evidence-grounded SFT data from Hugging Face.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8_000)
    parser.add_argument("--max-context-chars", type=int, default=320)
    parser.add_argument("--progress-every", type=int, default=1_000)
    args = parser.parse_args()
    if args.limit <= 0 or args.progress_every <= 0 or args.max_context_chars <= 0:
        parser.error("limit, context length, and progress interval must be positive")

    records: list[dict[str, object]] = []
    skipped = 0
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    for row in dataset:
        record = normalize_drcd_record(row, args.max_context_chars)
        if record is None:
            skipped += 1
            continue
        records.append(record)
        if len(records) % args.progress_every == 0:
            print(f"accepted {len(records):,} DRCD grounded examples", flush=True)
        if len(records) >= args.limit:
            break
    if not records:
        parser.error("no valid DRCD records found")
    write_jsonl(records, args.output)
    print(f"wrote {len(records):,} DRCD grounded SFT examples ({skipped:,} skipped) to {args.output}")


if __name__ == "__main__":
    main()
