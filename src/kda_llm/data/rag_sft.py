"""Build grounded SFT conversations from locally indexed reference chunks."""

from __future__ import annotations

import json
from pathlib import Path

from kda_llm.retrieval import load_index


QUESTION_TEMPLATES = (
    "請根據參考資料，簡要說明這個主題。",
    "請只依據參考資料回答：這段內容的重點是什麼？",
    "根據參考資料，請以繁體中文整理關鍵資訊。",
    "請根據參考資料解釋相關的技術設計，不要加入資料外的推測。",
    "若只能使用參考資料，應如何回答這個主題？",
    "請將參考資料改寫成簡潔的說明。",
)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    boundary = max(text.rfind("。", 0, limit), text.rfind("\n", 0, limit), text.rfind(".", 0, limit))
    return text[: boundary + 1 if boundary >= limit // 2 else limit].rstrip()


def _answer_from_context(context: str, limit: int) -> str:
    """Keep source facts while removing Markdown scaffolding from answer targets."""
    lines = [line.strip().lstrip("- ") for line in context.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return _truncate(" ".join(lines), limit)


def build_rag_sft_records(index_path: str, examples_per_chunk: int = 6, context_chars: int = 180, answer_chars: int = 180) -> list[dict[str, object]]:
    if examples_per_chunk <= 0 or context_chars <= 0 or answer_chars <= 0:
        raise ValueError("RAG-SFT sizes must be positive")
    records = []
    for chunk in load_index(index_path):
        context = _truncate(chunk["text"], context_chars)
        answer = _answer_from_context(context, answer_chars)
        if not context or not answer:
            continue
        system = (
            "僅根據下列參考資料回答。若資料不足，應回答不知道，"
            "不要捏造資料中沒有的事實。\n\n"
            f"參考資料：\n[1] 來源：{chunk['source']}\n{context}"
        )
        for index in range(examples_per_chunk):
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]},
                        {"role": "assistant", "content": answer},
                    ],
                    "source": chunk["source"],
                    "kind": "rag_sft",
                }
            )
    if not records:
        raise ValueError("RAG index contains no usable chunks")
    return records


def write_rag_sft_jsonl(records: list[dict[str, object]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
