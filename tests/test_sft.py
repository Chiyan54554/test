import torch

from kda_llm.training.sft import next_token_batch


def test_sft_labels_are_shifted_to_the_next_token() -> None:
    input_ids = torch.tensor([[11, 12, 13, 14]])
    labels = torch.tensor([[-100, -100, 13, 14]])

    x, y = next_token_batch(input_ids, labels)

    torch.testing.assert_close(x, torch.tensor([[11, 12, 13]]))
    torch.testing.assert_close(y, torch.tensor([[-100, 13, 14]]))
