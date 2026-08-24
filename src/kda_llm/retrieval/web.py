"""Brave Search API client for opt-in web-grounded generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str


def parse_brave_results(payload: dict[str, object]) -> list[WebHit]:
    web = payload.get("web")
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        return []
    hits = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title, url, snippet = result.get("title"), result.get("url"), result.get("description")
        if isinstance(title, str) and isinstance(url, str) and isinstance(snippet, str):
            hits.append(WebHit(title.strip(), url.strip(), snippet.strip()))
    return hits


def search_brave(query: str, api_key: str, count: int = 3, country: str = "TW", search_language: str = "zh-hant") -> list[WebHit]:
    if not api_key:
        raise ValueError("set BRAVE_SEARCH_API_KEY before using --web-search")
    if not query.strip() or not 1 <= count <= 20:
        raise ValueError("web query must be non-empty and count must be between 1 and 20")
    params = urlencode({"q": query, "count": count, "country": country, "search_lang": search_language})
    request = Request(f"https://api.search.brave.com/res/v1/web/search?{params}", headers={"Accept": "application/json", "X-Subscription-Token": api_key})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Brave Search API returned HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Brave Search API request failed: {error.reason}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Brave Search API returned an invalid response")
    return parse_brave_results(payload)


def render_web_context(hits: list[WebHit], max_chars: int = 256) -> str:
    sections, remaining = [], max_chars
    for index, hit in enumerate(hits, start=1):
        header = f"[網路 {index}] {hit.title}\nURL: {hit.url}\n摘要："
        budget = remaining - len(header)
        if budget <= 0:
            break
        snippet = hit.snippet[:budget].rstrip()
        sections.append(header + snippet)
        remaining -= len(header) + len(snippet)
    return "\n\n".join(sections)
