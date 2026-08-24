import os
from pathlib import Path
from tempfile import TemporaryDirectory

from kda_llm.env import load_env


def test_load_env_preserves_existing_environment() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / ".env"
        path.write_text("TEST_KDA_ENV=loaded\nEXISTING_KDA_ENV=file\n", encoding="utf-8")
        os.environ.pop("TEST_KDA_ENV", None)
        os.environ["EXISTING_KDA_ENV"] = "shell"
        load_env(str(path))
        assert os.environ["TEST_KDA_ENV"] == "loaded"
        assert os.environ["EXISTING_KDA_ENV"] == "shell"
        os.environ.pop("TEST_KDA_ENV", None)
        os.environ.pop("EXISTING_KDA_ENV", None)
