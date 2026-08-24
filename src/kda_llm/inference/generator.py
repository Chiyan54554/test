"""Checkpoint loading and cache-aware KDA text generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import sentencepiece as spm
import torch

from kda_llm.models import KDAConfig, KDALanguageModel

from .sampling import sample_next_token


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int = 42


def load_model(checkpoint_path: str, device: torch.device) -> KDALanguageModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict) or "model" not in checkpoint:
        raise ValueError("checkpoint is not a kda-train checkpoint")
    model = KDALanguageModel(KDAConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def format_chat_prompt(prompt: str, system_prompt: str | None = None) -> str:
    """Render the literal role format used by the SFT encoder."""
    parts = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt.strip()}")
    parts.extend((f"<|user|>\n{prompt.strip()}", "<|assistant|>\n"))
    return "\n".join(parts)


def generate(model: KDALanguageModel, tokenizer: spm.SentencePieceProcessor, prompt: str, config: GenerationConfig, device: torch.device) -> str:
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    prompt_ids = tokenizer.encode(prompt, out_type=int)[-model.config.max_seq_len :]
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
    with torch.inference_mode(), autocast:
        logits, _, cache = model(token_ids, use_cache=True)
        position = token_ids.size(1)
        for index in range(config.max_new_tokens):
            history = torch.cat((token_ids, torch.tensor([generated_ids], device=device)), dim=1) if generated_ids else token_ids
            next_token = sample_next_token(logits[:, -1], config.temperature, config.top_k, config.top_p, history, config.repetition_penalty)
            if next_token.item() == tokenizer.eos_id():
                break
            generated_ids.append(next_token.item())
            if index + 1 < config.max_new_tokens:
                logits, _, cache = model(next_token, past_states=cache, use_cache=True, position_offset=position)
                position += 1
    return tokenizer.decode(generated_ids)
