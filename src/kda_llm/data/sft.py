"""Streaming SFT ingestion and answer-only token encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def load_sources(path: str) -> list[dict[str, object]]:
    sources = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError("SFT source manifest must be a non-empty JSON array")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict) or not isinstance(source.get("dataset"), str):
            raise ValueError(f"source {index} must include a dataset")
        if source.get("format", "alpaca") not in {"alpaca", "conversations"}:
            raise ValueError(f"source {index} format must be alpaca or conversations")
    return sources


def normalize_messages(row: dict[str, object], source: dict[str, object]) -> list[dict[str, str]] | None:
    if source.get("format", "alpaca") == "conversations":
        raw_messages = row.get(str(source.get("messages_column", "conversations")))
        if not isinstance(raw_messages, list):
            return None
        messages = [{"role": str(item.get("role", "")), "content": str(item.get("content", "")).strip()} for item in raw_messages if isinstance(item, dict)]
    else:
        instruction = row.get(str(source.get("instruction_column", "instruction")))
        output = row.get(str(source.get("output_column", "output")))
        if not isinstance(instruction, str) or not isinstance(output, str):
            return None
        extra = row.get(str(source.get("input_column", "input")), "")
        user = instruction.strip() + ("\n\n" + extra.strip() if isinstance(extra, str) and extra.strip() else "")
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": output.strip()}]
    messages = [message for message in messages if message["role"] in {"system", "user", "assistant"} and message["content"]]
    return messages if len(messages) >= 2 and messages[-1]["role"] == "assistant" else None


def write_jsonl(records: list[dict[str, object]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def conversation_token_ids(tokenizer: object, messages: list[dict[str, str]], max_length: int) -> tuple[list[int], list[int]] | None:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        prefix = f"<|{role}|>\n"
        prefix_ids = tokenizer.encode(prefix, out_type=int)
        content_ids = tokenizer.encode(message["content"], out_type=int)
        input_ids.extend(prefix_ids + content_ids)
        labels.extend([-100] * len(prefix_ids) + (content_ids if role == "assistant" else [-100] * len(content_ids)))
        if role == "assistant":
            input_ids.append(tokenizer.eos_id())
            labels.append(tokenizer.eos_id())
    if len(input_ids) > max_length or not any(label != -100 for label in labels):
        return None
    return input_ids, labels


def encode_jsonl(input_path: str, tokenizer: object, max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    inputs, labels, skipped = [], [], 0
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            encoded = conversation_token_ids(tokenizer, record["messages"], max_length)
            if encoded is None:
                skipped += 1
                continue
            token_ids, target_ids = encoded
            inputs.append(token_ids + [tokenizer.pad_id()] * (max_length - len(token_ids)))
            labels.append(target_ids + [-100] * (max_length - len(target_ids)))
    if not inputs:
        raise ValueError("no SFT examples fit the requested max length")
    return torch.tensor(inputs, dtype=torch.uint16), torch.tensor(labels, dtype=torch.int16), skipped


def record_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.blake2b(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()
