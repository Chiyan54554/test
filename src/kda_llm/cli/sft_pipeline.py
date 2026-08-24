"""Compatibility entry point for the one-command SFT workflow."""

from kda_llm.workflows.sft_pipeline import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
