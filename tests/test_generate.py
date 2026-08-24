import torch

from kda_llm.cli.generate import sample_next_token
from kda_llm.inference import format_chat_prompt


def test_sample_next_token_returns_a_vocab_index() -> None:
    torch.manual_seed(7)
    token = sample_next_token(torch.tensor([[0.0, 1.0, 2.0]]), temperature=1.0, top_k=2, top_p=1.0)
    assert token.shape == (1, 1)
    assert token.item() in (1, 2)


def test_chat_prompt_matches_the_sft_role_format() -> None:
    assert format_chat_prompt("你好", "請使用繁體中文") == "<|system|>\n請使用繁體中文\n<|user|>\n你好\n<|assistant|>\n"
