import pytest
import torch

from phi3_pooling.pooling import LearnedLayerMix, PoolingHead, last_token_pool, masked_mean_pool


def test_last_token_pool_uses_last_valid_position() -> None:
    hidden = torch.tensor(
        [
            [[1.0], [2.0], [99.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    result = last_token_pool(hidden, mask)
    assert torch.equal(result, torch.tensor([[2.0], [5.0]]))


def test_masked_mean_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    result = masked_mean_pool(hidden, mask)
    assert torch.equal(result, torch.tensor([[2.0]]))


def test_p2_is_equal_last_four_mix() -> None:
    states = tuple(torch.full((1, 2, 1), float(value)) for value in range(1, 6))
    mask = torch.ones((1, 2), dtype=torch.long)
    result = PoolingHead("p2")(states, mask)
    assert torch.equal(result, torch.tensor([[3.5]]))


def test_p5_starts_as_equal_mix() -> None:
    states = tuple(torch.full((1, 2, 1), float(value)) for value in range(1, 5))
    mix = LearnedLayerMix()
    result = mix(states)
    assert torch.equal(result, torch.full((1, 2, 1), 2.5))


def test_empty_sequence_is_rejected() -> None:
    hidden = torch.zeros((1, 2, 3))
    mask = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="valid token"):
        masked_mean_pool(hidden, mask)

