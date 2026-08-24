from kda_llm.retrieval.web import parse_brave_results, render_web_context


def test_brave_result_parsing_and_context() -> None:
    hits = parse_brave_results({"web": {"results": [{"title": "KDA", "url": "https://example.com/kda", "description": "KDA reference"}]}})
    assert hits[0].url == "https://example.com/kda"
    assert "URL: https://example.com/kda" in render_web_context(hits)
