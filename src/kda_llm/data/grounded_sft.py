"""Normalize evidence-based QA datasets into the project's RAG chat format."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .sft import record_hash, write_jsonl


REFUSAL_ANSWER = "參考資料未提供這項資訊，因此不知道。"
SYSTEM_PROMPT = (
    "僅根據下列參考資料回答。每個事實後以 [來源編號] 標示依據。"
    "若資料不足，應回答不知道，不要捏造資料中沒有的事實。"
)


def _message_content(messages: object, role: str) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"].strip()
    return None


def _split_drcd_prompt(prompt: str) -> tuple[str, str] | None:
    if not prompt.startswith("文章：") or "\n問題：" not in prompt:
        return None
    context, question = prompt[len("文章：") :].rsplit("\n問題：", maxsplit=1)
    context, question = context.strip(), question.strip()
    return (context, question) if context and question else None


def _evidence_window(context: str, answer: str, limit: int) -> str:
    """Keep an answer-bearing local passage small enough for fixed-length SFT."""
    if len(context) <= limit:
        return context
    position = context.find(answer) if answer else -1
    if position < 0:
        return context[:limit].rstrip()
    start = max(0, position - limit // 2)
    end = min(len(context), start + limit)
    start = max(0, end - limit)
    return context[start:end].strip()


def normalize_drcd_record(row: dict[str, object], max_context_chars: int = 320) -> dict[str, object] | None:
    """Convert DRCD's JSON answer schema to a cited conversational answer."""
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")
    prompt = _message_content(row.get("messages"), "user")
    response = _message_content(row.get("messages"), "assistant")
    if prompt is None or response is None:
        return None
    parsed_prompt = _split_drcd_prompt(prompt)
    if parsed_prompt is None:
        return None
    try:
        target = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(target, dict):
        return None
    answerable = target.get("answerable")
    answer = target.get("answer")
    if not isinstance(answerable, bool) or not isinstance(answer, str):
        return None
    context, question = parsed_prompt
    context = _evidence_window(context, answer.strip() if answerable else "", max_context_chars)
    completion = f"{answer.strip()} [1]" if answerable and answer.strip() else REFUSAL_ANSWER
    source_id = str(row.get("article_id", row.get("qid", "drcd")))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"參考資料：\n[1] 來源：DRCD/{source_id}\n{context}\n\n問題：{question}"},
            {"role": "assistant", "content": completion},
        ],
        "source": "steven0226/drcd-zhtw-extractive-qa-sft",
        "kind": "drcd_answerable" if answerable else "drcd_refusal",
        "qid": row.get("qid"),
    }


def merge_jsonl(inputs: Iterable[str], output_path: str) -> int:
    """Merge JSONL artifacts while dropping duplicate conversations."""
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for input_path in inputs:
        path = Path(input_path)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                messages = record.get("messages") if isinstance(record, dict) else None
                if not isinstance(messages, list):
                    continue
                digest = record_hash(messages)
                if digest in seen:
                    continue
                seen.add(digest)
                records.append(record)
    if not records:
        raise ValueError("no grounded SFT records to merge")
    write_jsonl(records, output_path)
    return len(records)
