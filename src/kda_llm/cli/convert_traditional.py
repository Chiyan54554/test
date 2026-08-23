"""Convert a UTF-8 Chinese corpus to Traditional Chinese with OpenCC."""

from __future__ import annotations

import argparse
from pathlib import Path

from opencc import OpenCC


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Chinese text to Traditional Chinese.")
    parser.add_argument("--input", required=True, help="source UTF-8 .txt corpus")
    parser.add_argument("--output", required=True, help="converted UTF-8 .txt corpus")
    parser.add_argument("--progress-every", type=int, default=1000, help="print progress every N lines")
    parser.add_argument(
        "--config",
        default="s2twp",
        choices=("s2t", "s2tw", "s2twp"),
        help="OpenCC conversion mode; s2twp uses Taiwan variants and phrases",
    )
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be a positive integer")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("--input and --output must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    converter = OpenCC(args.config)
    line_count = 0
    with input_path.open("r", encoding="utf-8") as input_file, temporary_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            output_file.write(converter.convert(line))
            line_count += 1
            if line_count % args.progress_every == 0:
                print(f"converted {line_count:,} lines", flush=True)

    temporary_path.replace(output_path)
    print(f"converted {line_count:,} lines to {output_path}")


if __name__ == "__main__":
    main()
