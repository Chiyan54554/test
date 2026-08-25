"""Conservative source-conflict and sentence-evidence checks for RAG output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .bm25 import RAGHit, render_cited_answer, tokenize


SENTENCE_PATTERN = re.compile(r"(?<=[。！？.!?])\s*")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|億|萬|年|GB|MB|tokens?)?\b", re.IGNORECASE)


@dataclass(frozen=True)
class SourceConflict:
    left_source: str
    right_source: str
    left_claim: str
    right_claim: str


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]


def _content_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if len(term) >= 2}


def detect_source_conflicts(hits: list[RAGHit]) -> list[SourceConflict]:
    """Flag overlapping claims from different sources that state different numbers."""
    claims = [(hit.source, sentence, set(NUMBER_PATTERN.findall(sentence)), _content_terms(sentence)) for hit in hits for sentence in _sentences(hit.text)]
    conflicts = []
    for left_index, left in enumerate(claims):
        for right in claims[left_index + 1 :]:
            if left[0] == right[0] or not left[2] or not right[2] or left[2] == right[2]:
                continue
            if len(left[3].intersection(right[3])) >= 3:
                conflicts.append(SourceConflict(left[0], right[0], left[1], right[1]))
    return conflicts


def verify_answer_sentences(answer: str, hits: list[RAGHit], minimum_overlap: float = 0.5) -> list[tuple[str, list[int]]]:
    """Keep only answer sentences whose meaningful tokens are covered by a source."""
    if not 0 < minimum_overlap <= 1:
        raise ValueError("verification overlap must be in (0, 1]")
    verified = []
    for sentence in _sentences(answer):
        terms = _content_terms(sentence)
        if not terms:
            continue
        citations = [index for index, hit in enumerate(hits, start=1) if len(terms.intersection(_content_terms(hit.text))) / len(terms) >= minimum_overlap]
        if citations:
            verified.append((sentence, citations))
    return verified


def render_verified_answer(answer: str, hits: list[RAGHit], minimum_overlap: float = 0.5) -> str:
    verified = verify_answer_sentences(answer, hits, minimum_overlap)
    if not verified:
        return "資料不足，無法根據目前來源可靠回答。"
    return "".join(f"{sentence} {' '.join(f'[{index}]' for index in citations)}" for sentence, citations in verified)


def has_direct_query_evidence(query: str, hits: list[RAGHit], minimum_terms: int = 2) -> bool:
    """Require more than a shared topic name before replacing a refusal with evidence."""
    query_terms = _content_terms(query)
    return bool(query_terms) and any(len(query_terms.intersection(_content_terms(hit.text))) >= minimum_terms for hit in hits)


def render_reliable_answer(answer: str, query: str, hits: list[RAGHit], max_chars: int, minimum_overlap: float = 0.5) -> str:
    """Use model prose only when it is grounded; otherwise prefer direct evidence or refuse."""
    verified = render_verified_answer(answer, hits, minimum_overlap)
    model_refused = "不知道" in answer or "無法根據目前來源可靠回答" in verified
    too_short = len(_content_terms(answer)) < 2
    unreliable = model_refused or too_short or "無法根據目前來源可靠回答" in verified
    if unreliable:
        if has_direct_query_evidence(query, hits):
            return render_cited_answer(hits, query, max_chars)
        return "資料不足，無法根據目前來源可靠回答。"
    return verified
