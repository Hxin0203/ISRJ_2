"""Small deployable routers over two frozen recovery experts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ExpertRouter(nn.Module):
    def __init__(self, actions: int, hidden: int = 32) -> None:
        super().__init__()
        if actions not in (2, 3):
            raise ValueError("router actions must be 2 or 3")
        self.actions = actions
        self.network = nn.Sequential(
            nn.Conv1d(9, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, actions, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        if actions == 3:
            with torch.no_grad():
                self.network[-1].bias.copy_(torch.tensor([2.0, -2.0, -2.0]))
        else:
            nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def features(y: Tensor, control: Tensor, internal: Tensor, r_control: Tensor, r_internal: Tensor) -> Tensor:
        dc, di = control - y, internal - y
        disagreement = (control - internal).square().sum(1, keepdim=True).add(1e-12).sqrt()
        return torch.cat((y, dc, di, r_control, r_internal, disagreement), 1)

    def forward(self, y: Tensor, control: Tensor, internal: Tensor, r_control: Tensor, r_internal: Tensor) -> Tensor:
        return self.network(self.features(y, control, internal, r_control, r_internal))


def normalized_action_costs(y: Tensor, u: Tensor, control: Tensor, internal: Tensor, actions: int) -> Tensor:
    candidates = (control, internal) if actions == 2 else (y, control, internal)
    scale = u.square().sum(1, keepdim=True).add(1e-3)
    return torch.cat([((candidate - u).square().sum(1, keepdim=True) / scale).clamp_max(100.0) for candidate in candidates], 1)


def mix_actions(probability: Tensor, y: Tensor, control: Tensor, internal: Tensor) -> Tensor:
    candidates = (control, internal) if probability.shape[1] == 2 else (y, control, internal)
    return sum(probability[:, index:index + 1] * candidate for index, candidate in enumerate(candidates))


__all__ = ["ExpertRouter", "mix_actions", "normalized_action_costs"]
