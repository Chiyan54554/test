import torch

from kda_llm.model import KDALanguageModel, parameter_count


def test_model_outputs_next_token_logits() -> None:
    model = KDALanguageModel()
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    logits, loss = model(tokens[:, :-1], tokens[:, 1:])

    assert logits.shape == (2, 7, model.config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_model_stays_near_32m_parameters() -> None:
    model = KDALanguageModel()
    assert parameter_count(model) == 32_167_716


def test_cached_generation_matches_a_full_forward_pass() -> None:
    model = KDALanguageModel().eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 5))
    full_logits, _ = model(tokens)
    _, _, cache = model(tokens[:, :4], use_cache=True)
    cached_logits, _, _ = model(tokens[:, 4:], past_states=cache, use_cache=True, position_offset=4)
    torch.testing.assert_close(full_logits[:, 4:], cached_logits, rtol=1e-4, atol=1e-5)
