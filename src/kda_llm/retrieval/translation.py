"""Optional translation of web evidence before it reaches the small KDA model."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .web import WebHit


DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"


def translate_texts_to_traditional_chinese(
    texts: list[str],
    device: torch.device,
    model_name: str = DEFAULT_TRANSLATION_MODEL,
) -> list[str]:
    """Translate English evidence with NLLB without making it a base dependency."""
    if not texts:
        return []
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("web translation requires `uv sync --extra translation`") from error

    kwargs: dict[str, object] = {}
    if device.type == "cuda":
        kwargs["dtype"] = torch.float16
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(device).eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), 4):
                encoded = tokenizer(texts[start : start + 4], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                generated = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hant"), max_new_tokens=384)
                batches.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return batches
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def translate_web_hits(hits: list[WebHit], translate: Callable[[list[str]], list[str]]) -> list[WebHit]:
    """Keep source titles and URLs intact while replacing evidence snippets."""
    translated = translate([hit.snippet for hit in hits])
    if len(translated) != len(hits):
        raise RuntimeError("translation returned an unexpected number of web snippets")
    return [WebHit(hit.title, hit.url, snippet.strip() or hit.snippet) for hit, snippet in zip(hits, translated, strict=True)]
