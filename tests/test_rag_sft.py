import json
from pathlib import Path
from tempfile import TemporaryDirectory

from kda_llm.data.rag_sft import build_rag_sft_records, write_rag_sft_jsonl
from kda_llm.data.grounded_sft import REFUSAL_ANSWER, normalize_drcd_record


def test_rag_sft_records_include_context_and_answer() -> None:
    with TemporaryDirectory() as directory:
        index = Path(directory) / "index.json"
        index.write_text(json.dumps({"version": 1, "chunks": [{"source": "kda.md", "text": "# KDA\nKDA 使用 recurrent state 與 chunkwise kernel。"}]}), encoding="utf-8")

        records = build_rag_sft_records(str(index), examples_per_chunk=7)
        output = Path(directory) / "rag_sft.jsonl"
        write_rag_sft_jsonl(records, str(output))

        assert len(records) == 9
        assert "參考資料" in records[0]["messages"][0]["content"]
        assert records[0]["messages"][-1]["content"] == "KDA 使用 recurrent state 與 chunkwise kernel。 [1]"
        assert "KDA" in records[0]["messages"][1]["content"]
        assert records[6]["kind"] == "rag_fact"
        assert records[6]["messages"][-1]["content"].endswith("[1]")
        assert records[-1]["kind"] == "rag_refusal"
        assert len(output.read_text(encoding="utf-8").splitlines()) == 9


def test_drcd_records_are_converted_to_cited_answers_and_refusals() -> None:
    base = {
        "article_id": "42",
        "messages": [
            {"role": "user", "content": "文章：KDA 使用 recurrent state。\n問題：KDA 使用什麼？"},
            {"role": "assistant", "content": '{"answer": "recurrent state", "answerable": true}'},
        ],
    }
    answerable = normalize_drcd_record(base)
    assert answerable is not None
    assert "參考資料" in answerable["messages"][0]["content"]
    assert answerable["messages"][1]["content"] == "KDA 使用什麼？"
    assert answerable["messages"][-1]["content"] == "recurrent state [1]"

    base["messages"][-1]["content"] = '{"answer": "", "answerable": false}'
    refusal = normalize_drcd_record(base)
    assert refusal is not None
    assert refusal["messages"][-1]["content"] == REFUSAL_ANSWER


def test_drcd_answer_window_keeps_the_answer() -> None:
    row = {
        "messages": [
            {"role": "user", "content": f"文章：{'前' * 100}正確答案{'後' * 100}\n問題：答案是什麼？"},
            {"role": "assistant", "content": '{"answer": "正確答案", "answerable": true}'},
        ],
    }
    record = normalize_drcd_record(row, max_context_chars=40)
    assert record is not None
    assert "正確答案" in record["messages"][1]["content"]
