"""Stream a text column from a Hugging Face dataset into a UTF-8 corpus file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset


def load_sources(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict[str, object]]:
    if args.sources:
        manifest_path = Path(args.sources)
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            sources = json.load(manifest_file)
        if not isinstance(sources, list):
            parser.error("--sources must point to a JSON array")
    elif args.dataset:
        sources = [{
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "text_column": args.text_column,
            "limit": args.limit,
        }]
    else:
        parser.error("provide either --dataset or --sources")

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict) or not isinstance(source.get("dataset"), str):
            parser.error(f"source {index} must include a string dataset field")
        source.setdefault("split", "train")
        source.setdefault("text_column", "text")
        if not isinstance(source["split"], str) or not isinstance(source["text_column"], str):
            parser.error(f"source {index} split and text_column must be strings")
        limit = source.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            parser.error(f"source {index} limit must be a positive integer")
    return sources


def allocate_document_limits(
    sources: list[dict[str, object]], total_documents: int | None, parser: argparse.ArgumentParser
) -> None:
    if total_documents is None:
        return
    if any(source.get("limit") is not None for source in sources):
        parser.error("--total-documents cannot be combined with source limit values")
    if len(sources) == 1:
        sources[0]["limit"] = total_documents
        return

    weights = []
    for index, source in enumerate(sources, start=1):
        weight = source.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight) or weight <= 0:
            parser.error(f"source {index} needs a positive finite weight with --total-documents")
        weights.append(float(weight))

    normalized = [weight / sum(weights) for weight in weights]
    raw_limits = [total_documents * weight for weight in normalized]
    limits = [math.floor(limit) for limit in raw_limits]
    remainder_order = sorted(range(len(sources)), key=lambda index: raw_limits[index] - limits[index], reverse=True)
    for index in remainder_order[: total_documents - sum(limits)]:
        limits[index] += 1
    for source, limit in zip(sources, limits, strict=True):
        source["limit"] = limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face text dataset.")
    parser.add_argument("--dataset", help="dataset repository, for example org/dataset")
    parser.add_argument("--config", help="optional dataset configuration")
    parser.add_argument("--split", default="train", help="dataset split")
    parser.add_argument("--text-column", default="text", help="column containing plain text")
    parser.add_argument("--output", required=True, help="output UTF-8 .txt file")
    parser.add_argument("--limit", type=int, help="maximum documents to write")
    parser.add_argument("--sources", help="JSON array describing multiple dataset sources")
    parser.add_argument("--total-documents", type=int, help="total documents to download across all sources")
    args = parser.parse_args()
    if args.limit is not None and args.total_documents is not None:
        parser.error("--limit cannot be combined with --total-documents")
    if args.total_documents is not None and args.total_documents <= 0:
        parser.error("--total-documents must be a positive integer")
    sources = load_sources(args, parser)
    allocate_document_limits(sources, args.total_documents, parser)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_documents = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for source in sources:
            limit = source.get("limit")
            if limit == 0:
                print(f"{source['dataset']}: wrote 0 documents")
                continue
            dataset_args = {"path": source["dataset"], "split": source["split"], "streaming": True}
            if source.get("config"):
                dataset_args["name"] = source["config"]
            dataset = load_dataset(**dataset_args)
            document_count = 0
            for row in dataset:
                text = row.get(source["text_column"])
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                output_file.write(text + "\n")
                document_count += 1
                total_documents += 1
                if limit is not None and document_count >= limit:
                    break
            print(f"{source['dataset']}: wrote {document_count:,} documents")

    print(f"wrote {total_documents:,} documents to {output_path}")


if __name__ == "__main__":
    main()
