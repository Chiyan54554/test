import json

import kda_llm.retrieval.web as web
from kda_llm.retrieval.web import parse_brave_results, render_web_context


def test_brave_result_parsing_and_context() -> None:
    hits = parse_brave_results({"web": {"results": [{"title": "KDA", "url": "https://example.com/kda", "description": "KDA reference"}]}})
    assert hits[0].url == "https://example.com/kda"
    assert "URL: https://example.com/kda" in render_web_context(hits)


def test_wikipedia_result_parsing() -> None:
    original_get = web._get
    web._get = lambda _url: json.dumps({"pages": [{"title": "注意力機制", "excerpt": "<span>機器學習</span>方法"}]}).encode()
    try:
        hits = web.search_wikipedia("attention", 1)
    finally:
        web._get = original_get
    assert hits[0].title == "Wikipedia: 注意力機制"
    assert hits[0].snippet == "機器學習方法"
