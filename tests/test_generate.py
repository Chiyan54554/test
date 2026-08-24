import torch

from kda_llm.cli.generate import sample_next_token


def test_sample_next_token_returns_a_vocab_index() -> None:
    torch.manual_seed(7)
    token = sample_next_token(torch.tensor([[0.0, 1.0, 2.0]]), temperature=1.0, top_k=2, top_p=1.0)
    assert token.shape == (1, 1)
    assert token.item() in (1, 2)
