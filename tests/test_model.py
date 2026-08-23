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
