from kda_llm.retrieval import format_grounding_system


def test_grounding_system_contains_instruction_and_references() -> None:
    prompt = format_grounding_system("[1] 來源：kda.md\nKDA 使用 recurrent state。")

    assert "請以繁體中文回答" in prompt
    assert "[來源編號]" in prompt
    assert "參考資料" in prompt
    assert "來源：kda.md" in prompt
