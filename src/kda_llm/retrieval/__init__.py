"""Local retrieval utilities for grounded KDA generation."""

from .bm25 import RAGHit, build_index, load_index, render_context, retrieve
from .web import WebHit, render_web_context, search_brave

__all__ = ["RAGHit", "WebHit", "build_index", "load_index", "render_context", "render_web_context", "retrieve", "search_brave"]
