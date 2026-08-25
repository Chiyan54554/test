"""Shared prompt formatting for evidence-grounded training and inference."""

from __future__ import annotations


GROUNDING_INSTRUCTION = (
    "請以繁體中文回答；英文專有名詞可保留。僅根據下列參考資料回答；"
    "資料不足時請明確回答不知道，不要補充未提供的事實。"
)


def format_grounding_system(references: str) -> str:
    """Render the system message used by both RAG-SFT and RAG generation."""
    return f"{GROUNDING_INSTRUCTION}\n\n參考資料：\n{references.strip()}"
