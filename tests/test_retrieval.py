from kda_llm.retrieval import render_context, retrieve


def test_bm25_returns_the_relevant_technical_chunk() -> None:
    chunks = [
        {"source": "kda.md", "text": "Kimi Delta Attention 是一種線性注意力機制，使用 chunkwise kernel 處理長序列。"},
        {"source": "other.md", "text": "SentencePiece 用於將中文文字編碼成 subword token。"},
    ]

    hits = retrieve(chunks, "Kimi Delta Attention 的 chunkwise kernel 是什麼？")

    assert hits[0].source == "kda.md"
    assert "來源：kda.md" in render_context(hits)
