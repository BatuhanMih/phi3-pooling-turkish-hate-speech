"""Pooling operations used by the P0, P1, P2 and P5 conditions."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _validate_mask(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [batch, sequence, hidden_size]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden and attention_mask dimensions do not match")

    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each sample must contain at least one valid token")
    return mask


def last_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the last valid token for each sequence (P0)."""

    mask = _validate_mask(hidden, attention_mask)
    positions = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last_positions = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_index = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_index, last_positions]


def masked_mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Average valid token representations and ignore padding (P1)."""

    mask = _validate_mask(hidden, attention_mask)
    weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
    token_sum = (hidden * weights).sum(dim=1)
    token_count = weights.sum(dim=1).clamp_min(1)
    return token_sum / token_count


def _last_four(hidden_states: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if len(hidden_states) < 4:
        raise ValueError("at least four hidden states are required")
    selected = tuple(hidden_states[-4:])
    if any(state.shape != selected[0].shape for state in selected[1:]):
        raise ValueError("the last four hidden states must have the same shape")
    return selected  # type: ignore[return-value]


def equal_layer_mix(hidden_states: Sequence[Tensor]) -> Tensor:
    """Combine the last four hidden layers with fixed 0.25 weights (P2)."""

    layers = _last_four(hidden_states)
    return torch.stack(layers, dim=0).mean(dim=0)


class LearnedLayerMix(nn.Module):
    """Learn a softmax-normalized mixture of the last four layers (P5)."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4))

    def forward(self, hidden_states: Sequence[Tensor]) -> Tensor:
        layers = torch.stack(_last_four(hidden_states), dim=0)
        weights = torch.softmax(self.logits, dim=0)
        shape = (4,) + (1,) * (layers.ndim - 1)
        return (layers * weights.view(shape)).sum(dim=0)


class PoolingHead(nn.Module):
    """Expose the four ablation conditions through a common interface."""

    MODES = {"p0", "p1", "p2", "p5"}

    def __init__(self, mode: str) -> None:
        super().__init__()
        normalized = mode.lower()
        if normalized not in self.MODES:
            raise ValueError(f"unknown pooling mode: {mode}")
        self.mode = normalized
        self.layer_mix = LearnedLayerMix() if normalized == "p5" else None

    def forward(
        self,
        hidden_states: Sequence[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        if not hidden_states:
            raise ValueError("hidden_states cannot be empty")
        if self.mode == "p0":
            return last_token_pool(hidden_states[-1], attention_mask)
        if self.mode == "p1":
            return masked_mean_pool(hidden_states[-1], attention_mask)
        if self.mode == "p2":
            return masked_mean_pool(equal_layer_mix(hidden_states), attention_mask)
        assert self.layer_mix is not None
        return masked_mean_pool(self.layer_mix(hidden_states), attention_mask)

