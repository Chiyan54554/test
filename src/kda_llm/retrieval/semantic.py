"""Optional multilingual dense retrieval and cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bm25 import RAGHit, load_index


VECTOR_INDEX_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class VectorIndex:
    model_name: str
    chunks: list[dict[str, str]]
    embeddings: np.ndarray


def _transformers() -> tuple[object, object, object]:
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("semantic retrieval requires `uv sync --extra retrieval`") from error
    return AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _encode(texts: list[str], model_name: str, device: torch.device, prefix: str) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    AutoModel, _, AutoTokenizer = _transformers()
    kwargs: dict[str, object] = {"dtype": torch.float16} if device.type == "cuda" else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
    vectors = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                encoded = tokenizer([prefix + text for text in texts[start : start + 32]], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                output = model(**encoded)
                vectors.append(torch.nn.functional.normalize(_mean_pool(output.last_hidden_state, encoded["attention_mask"]), p=2, dim=1).float().cpu().numpy())
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(vectors, axis=0)


def build_vector_index(index_path: str, output_path: str, model_name: str, device: torch.device) -> int:
    chunks = load_index(index_path)
    embeddings = _encode([chunk["text"] for chunk in chunks], model_name, device, "passage: ")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, version=np.array([VECTOR_INDEX_VERSION]), model=np.array([model_name]), sources=np.array([chunk["source"] for chunk in chunks]), texts=np.array([chunk["text"] for chunk in chunks]), embeddings=embeddings)
    temporary.replace(output)
    return len(chunks)


def load_vector_index(path: str) -> VectorIndex:
    with np.load(path, allow_pickle=False) as payload:
        version = payload["version"].item()
        model_name = payload["model"].item()
        sources, texts, embeddings = payload["sources"], payload["texts"], payload["embeddings"]
    if version != VECTOR_INDEX_VERSION or not isinstance(model_name, str) or embeddings.ndim != 2 or len(sources) != len(texts) or len(texts) != len(embeddings):
        raise ValueError("invalid or unsupported vector RAG index")
    return VectorIndex(model_name, [{"source": str(source), "text": str(text)} for source, text in zip(sources, texts, strict=True)], embeddings.astype(np.float32, copy=False))


def vector_retrieve(index: VectorIndex, query: str, top_k: int, device: torch.device) -> list[RAGHit]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_embedding = _encode([query], index.model_name, device, "query: ")[0]
    scores = index.embeddings @ query_embedding
    order = np.argsort(-scores)[:top_k]
    return [RAGHit(index.chunks[position]["source"], index.chunks[position]["text"], float(scores[position])) for position in order]


def reciprocal_rank_fusion(rankings: list[list[RAGHit]], top_k: int, constant: int = 60) -> list[RAGHit]:
    """Fuse lexical and semantic rankings while preserving source text."""
    scores: dict[tuple[str, str], float] = {}
    hits: dict[tuple[str, str], RAGHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.text)
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
            hits[key] = hit
    return [RAGHit(hits[key].source, hits[key].text, score) for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def rerank(query: str, hits: list[RAGHit], model_name: str, device: torch.device) -> list[RAGHit]:
    if not hits:
        return []
    _, AutoModelForSequenceClassification, AutoTokenizer = _transformers()
    kwargs: dict[str, object] = {"dtype": torch.float16} if device.type == "cuda" else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to(device).eval()
    try:
        scores = []
        with torch.inference_mode():
            for start in range(0, len(hits), 16):
                batch = hits[start : start + 16]
                encoded = tokenizer([query] * len(batch), [hit.text for hit in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                scores.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return [RAGHit(hit.source, hit.text, score) for hit, score in sorted(zip(hits, scores, strict=True), key=lambda item: item[1], reverse=True)]
