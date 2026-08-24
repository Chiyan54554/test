import json
from pathlib import Path
from tempfile import TemporaryDirectory

from kda_llm.data.rag_sft import build_rag_sft_records, write_rag_sft_jsonl


def test_rag_sft_records_include_context_and_answer() -> None:
    with TemporaryDirectory() as directory:
        index = Path(directory) / "index.json"
        index.write_text(json.dumps({"version": 1, "chunks": [{"source": "kda.md", "text": "KDA 使用 recurrent state 與 chunkwise kernel。"}]}), encoding="utf-8")

        records = build_rag_sft_records(str(index), examples_per_chunk=2)
        output = Path(directory) / "rag_sft.jsonl"
        write_rag_sft_jsonl(records, str(output))

        assert len(records) == 2
        assert "參考資料" in records[0]["messages"][0]["content"]
        assert records[0]["messages"][-1]["content"] == "KDA 使用 recurrent state 與 chunkwise kernel。"
        assert len(output.read_text(encoding="utf-8").splitlines()) == 2
