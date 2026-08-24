"""Compatibility entry point for the end-to-end KDA workflow."""

from kda_llm.workflows.pipeline import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
