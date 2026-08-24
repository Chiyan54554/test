"""Stream-clean a Chinese corpus before tokenizer training and pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF


def normalize_text(text: str) -> str:
    """Normalize a document while removing invisible control characters."""
    normalized = unicodedata.normalize("NFKC", text)
    characters = []
    for character in normalized:
        if character.isspace():
            characters.append(" ")
        elif unicodedata.category(character).startswith("C"):
            continue
        else:
            characters.append(character)
    return WHITESPACE_RE.sub(" ", "".join(characters)).strip()


def reject_reason(text: str, min_chars: int, max_chars: int, min_cjk_ratio: float, max_url_ratio: float, max_repeated_char: int) -> str | None:
    length = len(text)
    if length < min_chars:
        return "too_short"
    if length > max_chars:
        return "too_long"
    if sum(is_cjk(character) for character in text) / length < min_cjk_ratio:
        return "not_chinese_enough"
    if sum(len(match.group()) for match in URL_RE.finditer(text)) / length > max_url_ratio:
        return "url_heavy"
    repeated = 1
    for previous, current in zip(text, text[1:]):
        repeated = repeated + 1 if current == previous else 1
        if repeated > max_repeated_char:
            return "repeated_characters"
    return None


def write_json_atomically(path: Path, data: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".partial")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, filter, and exact-deduplicate a Chinese text corpus.")
    parser.add_argument("--input", required=True, help="source UTF-8 corpus; one document per line")
    parser.add_argument("--output", required=True, help="cleaned UTF-8 corpus output")
    parser.add_argument("--stats", help="optional JSON report path; defaults beside --output")
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.15)
    parser.add_argument("--max-url-ratio", type=float, default=0.30)
    parser.add_argument("--max-repeated-char", type=int, default=16)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=1_000)
    args = parser.parse_args()
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        parser.error("--min-chars must be positive and no larger than --max-chars")
    if not 0 <= args.min_cjk_ratio <= 1 or not 0 <= args.max_url_ratio <= 1:
        parser.error("ratio options must be between 0 and 1")
    if args.max_repeated_char < 2 or args.progress_every <= 0:
        parser.error("--max-repeated-char must be at least 2 and --progress-every must be positive")

    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("--input and --output must be different files")
    stats_path = Path(args.stats) if args.stats else output_path.with_suffix(output_path.suffix + ".stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    database_path = output_path.with_suffix(output_path.suffix + ".dedupe.sqlite3")
    counts: Counter[str] = Counter()
    accepted_characters = 0
    # A previous interrupted run must not influence this fresh corpus build.
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path) if args.dedupe else None
    if connection is not None:
        connection.execute("CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY)")

    try:
        with input_path.open("r", encoding="utf-8") as input_file, temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for line in input_file:
                counts["input_documents"] += 1
                text = normalize_text(line)
                reason = reject_reason(text, args.min_chars, args.max_chars, args.min_cjk_ratio, args.max_url_ratio, args.max_repeated_char)
                if reason:
                    counts[reason] += 1
                    continue
                if connection is not None:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    try:
                        connection.execute("INSERT INTO seen VALUES (?)", (digest,))
                    except sqlite3.IntegrityError:
                        counts["duplicate"] += 1
                        continue
                output_file.write(text + "\n")
                counts["accepted_documents"] += 1
                accepted_characters += len(text)
                if counts["input_documents"] % args.progress_every == 0:
                    print(f"cleaned {counts['input_documents']:,} documents ({counts['accepted_documents']:,} accepted)", flush=True)
        if connection is not None:
            connection.commit()
        if not counts["accepted_documents"]:
            raise RuntimeError("cleaning rejected every document; relax the filter thresholds")
        temporary_path.replace(output_path)
        write_json_atomically(stats_path, {
            "input": str(input_path), "output": str(output_path), "accepted_characters": accepted_characters,
            "average_accepted_characters": accepted_characters / counts["accepted_documents"],
            "filters": {"min_chars": args.min_chars, "max_chars": args.max_chars, "min_cjk_ratio": args.min_cjk_ratio, "max_url_ratio": args.max_url_ratio, "max_repeated_char": args.max_repeated_char, "dedupe": args.dedupe},
            "counts": dict(sorted(counts.items())),
        })
        print(f"wrote {counts['accepted_documents']:,} clean documents to {output_path}")
        print(f"wrote cleaning report to {stats_path}")
    finally:
        if connection is not None:
            connection.close()
        database_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
