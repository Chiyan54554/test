"""Command-line entry point for cache-aware KDA text generation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.env import load_env
from kda_llm.inference import GenerationConfig, format_chat_prompt, generate, load_model, sample_next_token
from kda_llm.retrieval import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, DEFAULT_TRANSLATION_MODEL, detect_source_conflicts, format_grounding_system, load_index, load_vector_index, reciprocal_rank_fusion, render_cited_answer, render_context, render_verified_answer, render_web_context, rerank, retrieve, search_brave, search_free_knowledge, translate_texts_to_traditional_chinese, translate_web_hits, vector_retrieve


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Generate Traditional Chinese text from a KDA checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="penalty >= 1 for tokens already in context")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=False, help="render the SFT user/assistant template")
    parser.add_argument("--system", help="optional system instruction used with --chat")
    parser.add_argument("--rag-index", help="local JSON index created by kda-build-rag")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-max-context-chars", type=int, default=256)
    parser.add_argument("--rag-min-score", type=float, default=1.0, help="minimum local BM25 evidence score; use 0 to disable")
    parser.add_argument("--vector-index", help=".npz index created by kda-build-vector-rag")
    parser.add_argument("--retrieval-mode", choices=("bm25", "vector", "hybrid"), default="bm25")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="used only to validate --vector-index model identity")
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=False, help="cross-encoder rerank retrieved candidates")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-candidates", type=int, default=20)
    parser.add_argument("--source-conflict", choices=("ignore", "warn", "refuse"), default="warn")
    parser.add_argument("--verification-min-overlap", type=float, default=0.5)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--rag-answer-mode", choices=("generate", "extractive", "cited", "verified"), default="generate", help="generate, return source text, return cited evidence, or verify generated sentences")
    parser.add_argument("--web-search", action="store_true", help="search arXiv and Chinese Wikipedia before answering")
    parser.add_argument("--web-provider", choices=("academic", "brave"), default="academic")
    parser.add_argument("--web-count", type=int, default=3)
    parser.add_argument("--web-country", default="TW")
    parser.add_argument("--web-language", default="zh-hant")
    parser.add_argument("--web-max-context-chars", type=int, default=256)
    parser.add_argument("--translate-web-sources", action=argparse.BooleanOptionalAction, default=True, help="translate English web evidence to Traditional Chinese with NLLB")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty < 1 or args.rag_top_k <= 0 or args.rag_max_context_chars <= 0 or args.rag_min_score < 0 or args.reranker_candidates <= 0 or not 0 < args.verification_min_overlap <= 1 or not 1 <= args.web_count <= 20 or args.web_max_context_chars <= 0:
        parser.error("invalid sampling options")
    if args.retrieval_mode != "bm25" and not args.vector_index:
        parser.error("--retrieval-mode vector or hybrid requires --vector-index")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    system_prompt = args.system
    hits, web_hits, contexts = [], [], []
    if args.rag_index:
        try:
            lexical_hits = retrieve(load_index(args.rag_index), args.prompt, max(args.rag_top_k, args.reranker_candidates))
            if args.rag_min_score and (not lexical_hits or lexical_hits[0].score < args.rag_min_score):
                best_score = lexical_hits[0].score if lexical_hits else 0.0
                parser.error(f"RAG evidence is too weak (best BM25 score {best_score:.2f} < {args.rag_min_score:.2f}); unable to answer reliably")
            if args.retrieval_mode == "bm25":
                hits = lexical_hits
            else:
                vector_index = load_vector_index(args.vector_index)
                if vector_index.model_name != args.embedding_model:
                    parser.error("--embedding-model must match the model stored in --vector-index")
                semantic_hits = vector_retrieve(vector_index, args.prompt, max(args.rag_top_k, args.reranker_candidates), device)
                hits = semantic_hits if args.retrieval_mode == "vector" else reciprocal_rank_fusion([lexical_hits, semantic_hits], args.reranker_candidates)
            if args.reranker:
                hits = rerank(args.prompt, hits[: args.reranker_candidates], args.reranker_model, device)
            hits = hits[: args.rag_top_k]
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        if not hits:
            parser.error("RAG found no relevant reference chunks")
        conflicts = detect_source_conflicts(hits)
        if conflicts and args.source_conflict != "ignore":
            message = f"source conflict: {conflicts[0].left_source} and {conflicts[0].right_source} report different overlapping numeric claims"
            if args.source_conflict == "refuse":
                parser.error(message)
            print(f"warning: {message}", file=sys.stderr)
        contexts.append(render_context(hits, args.rag_max_context_chars, args.prompt))
    if args.web_search:
        try:
            web_hits = search_free_knowledge(args.prompt, args.web_count) if args.web_provider == "academic" else search_brave(args.prompt, os.getenv("BRAVE_SEARCH_API_KEY", ""), args.web_count, args.web_country, args.web_language)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        if not web_hits:
            parser.error("web search found no usable result snippets")
        if args.translate_web_sources:
            try:
                web_hits = translate_web_hits(
                    web_hits,
                    lambda snippets: translate_texts_to_traditional_chinese(snippets, device, args.translation_model),
                )
            except RuntimeError as error:
                parser.error(str(error))
        contexts.append(render_web_context(web_hits, args.web_max_context_chars))
    if contexts:
        references = "\n\n".join(contexts)
        system_prompt = f"{system_prompt.strip()}\n\n" if system_prompt else ""
        system_prompt += format_grounding_system(references)
        args.chat = True
    rendered_prompt = format_chat_prompt(args.prompt, system_prompt) if args.chat else args.prompt
    if args.show_sources and hits:
        for index, hit in enumerate(hits, start=1):
            print(f"[{index}] {hit.source} (BM25 {hit.score:.2f})")
    if args.show_sources and web_hits:
        for index, hit in enumerate(web_hits, start=1):
            print(f"[網路 {index}] {hit.title} | {hit.url}")
    if args.rag_answer_mode in {"extractive", "cited"}:
        if not contexts:
            parser.error("non-generative RAG modes require --rag-index or --web-search")
        if args.rag_answer_mode == "cited":
            answers = [render_cited_answer(hits, args.prompt, args.rag_max_context_chars)] if hits else []
            answers.extend(render_web_context(web_hits, args.web_max_context_chars) for _ in [None] if web_hits)
            print("\n\n".join(answers))
        else:
            print("\n\n".join(contexts))
        return
    if args.rag_answer_mode == "verified" and not hits:
        parser.error("verified mode requires --rag-index for sentence evidence checks")
    model = load_model(args.checkpoint, device)
    if tokenizer.vocab_size() != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the checkpoint")
    completion = generate(model, tokenizer, rendered_prompt, GenerationConfig(args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.seed), device)
    completion = render_verified_answer(completion, hits, args.verification_min_overlap) if args.rag_answer_mode == "verified" else completion
    print(completion)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_prompt + completion + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
