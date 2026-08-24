"""Dependency-free BM25 retrieval for Chinese and technical reference files."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


INDEX_VERSION = 1
SUPPORTED_SUFFIXES = {".md", ".txt"}
WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.IGNORECASE)
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True)
class RAGHit:
    source: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    """Use English identifiers plus Chinese characters and bigrams for recall."""
    normalized = text.lower()
    tokens = WORD_PATTERN.findall(normalized)
    for sequence in CJK_PATTERN.findall(normalized):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]


def _source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("RAG input file must be .md or .txt")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def build_index(input_path: str, output_path: str, chunk_chars: int = 600, overlap_chars: int = 80) -> tuple[int, int]:
    if chunk_chars <= 0 or not 0 <= overlap_chars < chunk_chars:
        raise ValueError("chunk chars must be positive and overlap must be smaller than the chunk")
    root = Path(input_path)
    files = _source_files(root)
    if not files:
        raise ValueError("no .md or .txt files found for RAG indexing")
    chunks = []
    for path in files:
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        for text in _chunk_text(path.read_text(encoding="utf-8"), chunk_chars, overlap_chars):
            chunks.append({"source": source, "text": text})
    if not chunks:
        raise ValueError("reference files contain no indexable text")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps({"version": INDEX_VERSION, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return len(files), len(chunks)


def load_index(index_path: str) -> list[dict[str, str]]:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid or unsupported RAG index")
    chunks = payload.get("chunks")
    if payload.get("version") != INDEX_VERSION or not isinstance(chunks, list):
        raise ValueError("invalid or unsupported RAG index")
    if not all(isinstance(chunk, dict) and isinstance(chunk.get("source"), str) and isinstance(chunk.get("text"), str) for chunk in chunks):
        raise ValueError("RAG index contains malformed chunks")
    return chunks


def retrieve(chunks: list[dict[str, str]], query: str, top_k: int = 3) -> list[RAGHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_terms = tokenize(query)
    if not query_terms:
        return []
    documents = [Counter(tokenize(chunk["text"])) for chunk in chunks]
    document_frequency = Counter(term for document in documents for term in document)
    average_length = sum(sum(document.values()) for document in documents) / len(documents)
    scores = []
    for chunk, document in zip(chunks, documents, strict=True):
        length, score = sum(document.values()), 0.0
        for term in query_terms:
            frequency = document.get(term, 0)
            if not frequency:
                continue
            idf = math.log1p((len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * frequency * 2.2 / (frequency + 1.2 * (1 - 0.75 + 0.75 * length / average_length))
        if score > 0:
            scores.append(RAGHit(chunk["source"], chunk["text"], score))
    return sorted(scores, key=lambda hit: hit.score, reverse=True)[:top_k]


def render_context(hits: list[RAGHit], max_chars: int = 500) -> str:
    if max_chars <= 0:
        raise ValueError("max context chars must be positive")
    sections, remaining = [], max_chars
    for index, hit in enumerate(hits, start=1):
        header = f"[{index}] 來源：{hit.source}\n"
        budget = remaining - len(header)
        if budget <= 0:
            break
        snippet = hit.text[:budget].rstrip()
        sections.append(header + snippet)
        remaining -= len(header) + len(snippet)
        if remaining <= 0:
            break
    return "\n\n".join(sections)


def render_cited_answer(hits: list[RAGHit], query: str, max_chars: int = 500) -> str:
    """Return evidence sentences with inline source markers, without model generation."""
    if max_chars <= 0:
        raise ValueError("max chars must be positive")
    query_terms = set(tokenize(query))
    sections, remaining = [], max_chars
    for index, hit in enumerate(hits, start=1):
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])\s*", hit.text) if sentence.strip()]
        best = max(sentences, key=lambda sentence: len(query_terms.intersection(tokenize(sentence))), default=hit.text.strip())
        answer = f"{best} [{index}]"
        header = f"[{index}] 來源：{hit.source}\n"
        if len(header) >= remaining:
            break
        answer = answer[: remaining - len(header)].rstrip()
        if answer:
            sections.append(header + answer)
            remaining -= len(header) + len(answer)
        if remaining <= 0:
            break
    return "\n\n".join(sections)
