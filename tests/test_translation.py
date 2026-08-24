import pytest

from kda_llm.retrieval import WebHit, normalize_traditional_chinese, translate_web_hits


def test_normalize_traditional_chinese_converts_each_translation() -> None:
    assert normalize_traditional_chinese(["比较缓存"], lambda text: text.replace("比较", "比較").replace("缓存", "快取")) == ["比較快取"]


def test_translate_web_hits_only_replaces_snippets() -> None:
    hits = [WebHit("arXiv: KDA", "https://example.test/kda", "An English abstract.")]

    translated = translate_web_hits(hits, lambda snippets: [f"繁中：{snippet}" for snippet in snippets])

    assert translated == [WebHit("arXiv: KDA", "https://example.test/kda", "繁中：An English abstract.")]


def test_translate_web_hits_rejects_mismatched_result_count() -> None:
    with pytest.raises(RuntimeError, match="unexpected number"):
        translate_web_hits([WebHit("title", "https://example.test", "snippet")], lambda _: [])
