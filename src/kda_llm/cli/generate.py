"""Command-line entry point for cache-aware KDA text generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.env import load_env
from kda_llm.inference import GenerationConfig, format_chat_prompt, generate, load_model, sample_next_token
from kda_llm.retrieval import load_index, render_context, render_web_context, retrieve, search_brave, search_free_knowledge


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
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--rag-answer-mode", choices=("generate", "extractive"), default="generate", help="generate with context or return retrieved source text directly")
    parser.add_argument("--web-search", action="store_true", help="search arXiv and Chinese Wikipedia before answering")
    parser.add_argument("--web-provider", choices=("academic", "brave"), default="academic")
    parser.add_argument("--web-count", type=int, default=3)
    parser.add_argument("--web-country", default="TW")
    parser.add_argument("--web-language", default="zh-hant")
    parser.add_argument("--web-max-context-chars", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty < 1 or args.rag_top_k <= 0 or args.rag_max_context_chars <= 0 or not 1 <= args.web_count <= 20 or args.web_max_context_chars <= 0:
        parser.error("invalid sampling options")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.checkpoint, device)
    if tokenizer.vocab_size() != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the checkpoint")
    system_prompt = args.system
    hits, web_hits, contexts = [], [], []
    if args.rag_index:
        try:
            hits = retrieve(load_index(args.rag_index), args.prompt, args.rag_top_k)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if not hits:
            parser.error("RAG found no relevant reference chunks")
        contexts.append(render_context(hits, args.rag_max_context_chars))
    if args.web_search:
        try:
            web_hits = search_free_knowledge(args.prompt, args.web_count) if args.web_provider == "academic" else search_brave(args.prompt, os.getenv("BRAVE_SEARCH_API_KEY", ""), args.web_count, args.web_country, args.web_language)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        if not web_hits:
            parser.error("web search found no usable result snippets")
        contexts.append(render_web_context(web_hits, args.web_max_context_chars))
    if contexts:
        instruction = "請以繁體中文回答；英文專有名詞可保留。僅根據下列參考資料回答；資料不足時請明確回答不知道，不要補充未提供的事實。"
        references = "\n\n".join(contexts)
        system_prompt = f"{system_prompt.strip()}\n\n" if system_prompt else ""
        system_prompt += f"{instruction}\n\n參考資料：\n{references}"
        args.chat = True
    rendered_prompt = format_chat_prompt(args.prompt, system_prompt) if args.chat else args.prompt
    if args.show_sources and hits:
        for index, hit in enumerate(hits, start=1):
            print(f"[{index}] {hit.source} (BM25 {hit.score:.2f})")
    if args.show_sources and web_hits:
        for index, hit in enumerate(web_hits, start=1):
            print(f"[網路 {index}] {hit.title} | {hit.url}")
    if args.rag_answer_mode == "extractive":
        if not contexts:
            parser.error("extractive mode requires --rag-index or --web-search")
        print("\n\n".join(contexts))
        return
    completion = generate(model, tokenizer, rendered_prompt, GenerationConfig(args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.seed), device)
    print(completion)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_prompt + completion + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
