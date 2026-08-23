"""Convert a UTF-8 Chinese corpus to Traditional Chinese with OpenCC."""

from __future__ import annotations

import argparse
from pathlib import Path

from opencc import OpenCC


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Chinese text to Traditional Chinese.")
    parser.add_argument("--input", required=True, help="source UTF-8 .txt corpus")
    parser.add_argument("--output", required=True, help="converted UTF-8 .txt corpus")
    parser.add_argument(
        "--config",
        default="s2twp",
        choices=("s2t", "s2tw", "s2twp"),
        help="OpenCC conversion mode; s2twp uses Taiwan variants and phrases",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        parser.error("--input and --output must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = OpenCC(args.config)
    line_count = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            output_file.write(converter.convert(line))
            line_count += 1

    print(f"converted {line_count:,} lines to {output_path}")


if __name__ == "__main__":
    main()
