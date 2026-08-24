"""Local retrieval utilities for grounded KDA generation."""

from .bm25 import RAGHit, build_index, load_index, render_context, retrieve

__all__ = ["RAGHit", "build_index", "load_index", "render_context", "retrieve"]
