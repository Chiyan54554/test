"""Stream structured Traditional Chinese instruction data from Hugging Face."""

from __future__ import annotations

import argparse

from datasets import load_dataset

from kda_llm.data.sft import load_sources, normalize_messages, record_hash, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream SFT conversations from Hugging Face into JSONL.")
    parser.add_argument("--sources", required=True, help="SFT source manifest JSON")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=50_000, help="maximum accepted examples")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.limit <= 0 or args.progress_every <= 0:
        parser.error("limit and progress interval must be positive")
    records, seen = [], set()
    for source in load_sources(args.sources):
        if len(records) >= args.limit:
            break
        dataset_args = {"path": source["dataset"], "split": source.get("split", "train"), "streaming": True}
        if source.get("config"):
            dataset_args["name"] = source["config"]
        dataset = load_dataset(**dataset_args)
        source_limit = int(source.get("limit", args.limit))
        accepted = 0
        for row in dataset:
            messages = normalize_messages(row, source)
            if messages is None or record_hash(messages) in seen:
                continue
            seen.add(record_hash(messages))
            records.append({"messages": messages, "source": source["dataset"]})
            accepted += 1
            if len(records) % args.progress_every == 0:
                print(f"accepted {len(records):,} SFT examples", flush=True)
            if accepted >= source_limit or len(records) >= args.limit:
                break
        print(f"{source['dataset']}: accepted {accepted:,} examples", flush=True)
    write_jsonl(records, args.output)
    print(f"wrote {len(records):,} SFT examples to {args.output}")


if __name__ == "__main__":
    main()
