from kda_llm.retrieval import render_cited_answer, render_context, retrieve


def test_bm25_returns_the_relevant_technical_chunk() -> None:
    chunks = [
        {"source": "kda.md", "text": "Kimi Delta Attention 是一種線性注意力機制，使用 chunkwise kernel 處理長序列。"},
        {"source": "other.md", "text": "SentencePiece 用於將中文文字編碼成 subword token。"},
    ]

    hits = retrieve(chunks, "Kimi Delta Attention 的 chunkwise kernel 是什麼？")

    assert hits[0].source == "kda.md"
    assert "來源：kda.md" in render_context(hits)
    assert "[1]" in render_cited_answer(hits, "Kimi Delta Attention 的 chunkwise kernel 是什麼？")


def test_context_selects_a_relevant_sentence_inside_a_chunk() -> None:
    hits = [RAGHit("kda.md", "KDA 會處理輸入 token。KDA 維護 recurrent state，因此適合長序列。", 1.0)]

    context = render_context(hits, max_chars=60, query="KDA 為何適合長序列？")

    assert "recurrent state" in context
