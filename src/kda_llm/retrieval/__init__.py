"""Local retrieval utilities for grounded KDA generation."""

from .bm25 import RAGHit, build_index, load_index, render_cited_answer, render_context, retrieve
from .prompts import GROUNDING_INSTRUCTION, format_grounding_system
from .semantic import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, build_vector_index, load_vector_index, reciprocal_rank_fusion, rerank, vector_retrieve
from .translation import DEFAULT_TRANSLATION_MODEL, normalize_traditional_chinese, translate_texts_to_traditional_chinese, translate_web_hits
from .verification import SourceConflict, detect_source_conflicts, has_direct_query_evidence, render_reliable_answer, render_verified_answer, verify_answer_sentences
from .web import WebHit, render_web_context, search_brave, search_free_knowledge

__all__ = ["DEFAULT_EMBEDDING_MODEL", "DEFAULT_RERANKER_MODEL", "DEFAULT_TRANSLATION_MODEL", "GROUNDING_INSTRUCTION", "RAGHit", "SourceConflict", "WebHit", "build_index", "build_vector_index", "detect_source_conflicts", "format_grounding_system", "has_direct_query_evidence", "load_index", "load_vector_index", "normalize_traditional_chinese", "reciprocal_rank_fusion", "render_cited_answer", "render_context", "render_reliable_answer", "render_verified_answer", "render_web_context", "rerank", "retrieve", "search_brave", "search_free_knowledge", "translate_texts_to_traditional_chinese", "translate_web_hits", "vector_retrieve", "verify_answer_sentences"]
