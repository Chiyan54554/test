"""Compatibility entry point for the one-command RAG-SFT workflow."""

from kda_llm.workflows.rag_sft_pipeline import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
