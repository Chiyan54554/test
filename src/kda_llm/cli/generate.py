"""Command-line entry point for cache-aware KDA text generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from kda_llm.inference import GenerationConfig, format_chat_prompt, generate, load_model, sample_next_token
from kda_llm.retrieval import load_index, render_context, retrieve


def main() -> None:
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
    parser.add_argument("--rag-max-context-chars", type=int, default=500)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty < 1 or args.rag_top_k <= 0 or args.rag_max_context_chars <= 0:
        parser.error("invalid sampling options")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available to PyTorch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_model(args.checkpoint, device)
    if tokenizer.vocab_size() != model.config.vocab_size:
        parser.error("tokenizer vocabulary size does not match the checkpoint")
    system_prompt = args.system
    hits = []
    if args.rag_index:
        try:
            hits = retrieve(load_index(args.rag_index), args.prompt, args.rag_top_k)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if not hits:
            parser.error("RAG found no relevant reference chunks")
        context = render_context(hits, args.rag_max_context_chars)
        instruction = "僅根據下列參考資料回答；資料不足時請明確回答不知道，不要補充未提供的事實。"
        system_prompt = f"{system_prompt.strip()}\n\n" if system_prompt else ""
        system_prompt += f"{instruction}\n\n參考資料：\n{context}"
        args.chat = True
    rendered_prompt = format_chat_prompt(args.prompt, system_prompt) if args.chat else args.prompt
    if args.show_sources and hits:
        for index, hit in enumerate(hits, start=1):
            print(f"[{index}] {hit.source} (BM25 {hit.score:.2f})")
    completion = generate(model, tokenizer, rendered_prompt, GenerationConfig(args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, args.seed), device)
    print(completion)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_prompt + completion + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
