"""Build grounded SFT conversations from locally indexed reference chunks."""

from __future__ import annotations

import json
from pathlib import Path

from kda_llm.retrieval import format_grounding_system, load_index


QUESTION_TEMPLATES = (
    "請根據參考資料，簡要說明「{topic}」。",
    "請只依據參考資料回答：「{topic}」的重點是什麼？",
    "根據參考資料，請以繁體中文整理「{topic}」的關鍵資訊。",
    "請根據參考資料解釋「{topic}」的技術設計，不要加入資料外的推測。",
    "若只能使用參考資料，應如何回答「{topic}」？",
    "請將參考資料中關於「{topic}」的內容改寫成簡潔說明。",
)

FACT_QUESTION_TEMPLATES = (
    "請指出參考資料中一項關於「{topic}」可驗證的事實。",
    "根據參考資料，「{topic}」如何運作？請只回答資料明示的內容。",
    "參考資料提到「{topic}」的哪個特性？",
    "請用一句話整理參考資料對「{topic}」的描述。",
)

REFUSAL_TEMPLATES = (
    "參考資料沒有提供的資訊是什麼？請不要猜測。",
    "請列出參考資料未記載的硬體價格與規格。",
    "如果參考資料不足以回答，應如何回覆？",
)
REFUSAL_ANSWER = "參考資料未提供這項資訊，因此不知道。"


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


def _topic_from_chunk(text: str, source: str) -> str:
    """Use a document heading when available so questions are anchored to a real topic."""
    for line in text.splitlines():
        heading = line.strip().lstrip("#").strip()
        if heading:
            return heading[:80]
    return Path(source).stem.replace("_", " ") or "參考資料中的主題"


def _facts_from_context(context: str, limit: int) -> list[str]:
    """Split source prose into short, independently verifiable answer targets."""
    prose = _answer_from_context(context, limit * 4)
    facts = [_truncate(sentence, limit) for sentence in prose.replace("！", "。").replace("？", "。").split("。") if sentence.strip()]
    return [fact for fact in facts if fact]


def build_rag_sft_records(index_path: str, examples_per_chunk: int = 6, context_chars: int = 180, answer_chars: int = 180, refusal_ratio: float = 0.25, context_chunks: int = 1) -> list[dict[str, object]]:
    if examples_per_chunk <= 0 or context_chars <= 0 or answer_chars <= 0 or context_chunks <= 0:
        raise ValueError("RAG-SFT sizes must be positive")
    if not 0 <= refusal_ratio < 1:
        raise ValueError("refusal ratio must be in [0, 1)")
    chunks = load_index(index_path)
    records = []
    refusal_count = round(examples_per_chunk * refusal_ratio / max(1 - refusal_ratio, 1e-6))
    for chunk_index, chunk in enumerate(chunks):
        selected = [chunks[(chunk_index + offset) % len(chunks)] for offset in range(context_chunks)]
        budget = max(1, context_chars // len(selected))
        contexts = [_truncate(item["text"], budget) for item in selected]
        answers = [_answer_from_context(context, max(1, answer_chars // len(selected))) for context in contexts]
        if not all(contexts) or not all(answers):
            continue
        references = "\n\n".join(f"[{index}] 來源：{item['source']}\n{context}" for index, (item, context) in enumerate(zip(selected, contexts, strict=True), start=1))
        answer = " ".join(f"{text} [{index}]" for index, text in enumerate(answers, start=1))
        system = format_grounding_system(references)
        topic = _topic_from_chunk(chunk["text"], chunk["source"])
        full_answer = " ".join(f"{text} [{index}]" for index, text in enumerate(answers, start=1))
        facts = _facts_from_context(contexts[0], answer_chars)
        if not facts:
            facts = [answers[0]]
        for index in range(examples_per_chunk):
            if index < len(QUESTION_TEMPLATES):
                question = QUESTION_TEMPLATES[index].format(topic=topic)
                answer = full_answer
                kind = "rag_summary"
            else:
                fact_index = index - len(QUESTION_TEMPLATES)
                question = FACT_QUESTION_TEMPLATES[fact_index % len(FACT_QUESTION_TEMPLATES)].format(topic=topic)
                answer = f"{facts[fact_index % len(facts)]} [1]"
                kind = "rag_fact"
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "source": chunk["source"],
                    "kind": kind,
                }
            )
        for index in range(refusal_count):
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": REFUSAL_TEMPLATES[index % len(REFUSAL_TEMPLATES)]},
                        {"role": "assistant", "content": REFUSAL_ANSWER},
                    ],
                    "source": chunk["source"],
                    "kind": "rag_refusal",
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
