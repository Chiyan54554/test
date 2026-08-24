from kda_llm.retrieval import RAGHit, detect_source_conflicts, reciprocal_rank_fusion, render_verified_answer


def test_hybrid_rank_fusion_rewards_a_chunk_found_by_both_retrievers() -> None:
    shared = RAGHit("kda.md", "KDA uses recurrent state.", 0.8)
    fused = reciprocal_rank_fusion([[shared, RAGHit("other.md", "Other text.", 0.7)], [shared]])

    assert fused[0].source == "kda.md"


def test_conflict_detector_flags_different_numeric_claims_from_sources() -> None:
    hits = [
        RAGHit("a.md", "KDA 的 context 長度為 128 tokens。", 2.0),
        RAGHit("b.md", "KDA 的 context 長度為 256 tokens。", 1.9),
    ]

    assert detect_source_conflicts(hits)


def test_verified_answer_removes_unsupported_sentence() -> None:
    hits = [RAGHit("kda.md", "KDA 使用 recurrent state 處理長序列。", 2.0)]

    answer = render_verified_answer("KDA 使用 recurrent state 處理長序列。KDA 的作者住在台北。", hits)

    assert "recurrent state" in answer
    assert "作者住在台北" not in answer
    assert "[1]" in answer
