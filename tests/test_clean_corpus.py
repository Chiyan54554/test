from kda_llm.cli.clean_corpus import normalize_text, reject_reason


def test_normalize_text_removes_controls_and_collapses_whitespace() -> None:
    assert normalize_text("  臺灣\x00\n 語言　模型  ") == "臺灣 語言 模型"


def test_reject_reason_filters_low_quality_documents() -> None:
    assert reject_reason("太短", 20, 100, 0.15, 0.3, 16) == "too_short"
    assert reject_reason("x" * 30, 20, 100, 0.15, 0.3, 16) == "not_chinese_enough"
    assert reject_reason("中" * 17 + "文內容足夠長", 20, 100, 0.15, 0.3, 16) == "repeated_characters"


def test_reject_reason_accepts_normal_chinese_text() -> None:
    text = "這是一段內容完整的繁體中文文件，用於測試語料清理流程是否保留正常資料。"
    assert reject_reason(text, 20, 100, 0.15, 0.3, 16) is None
