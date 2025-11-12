"""Action space implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Sequence

from gymnasium import spaces

from rl.registries import register_action


class BaseActionSpace:
    """Base class for action-space adapters."""

    version: str = "base"

    def gym_space(self) -> spaces.Space:  # pragma: no cover - interface only
        raise NotImplementedError

    def n(self) -> int:  # pragma: no cover - interface only
        raise NotImplementedError

    def id_to_action(self, action_idx: int) -> tuple[int, int]:  # pragma: no cover - interface only
        raise NotImplementedError

    def action_to_id(self, destroy_idx: int, repair_idx: int) -> int:  # pragma: no cover - interface only
        raise NotImplementedError

    def sample(self) -> int:  # pragma: no cover - interface only
        raise NotImplementedError


@register_action("op_pair_v1")
class OperatorPairActionSpace(BaseActionSpace):
    """Map discrete indices to destroy/repair operator pairs."""

    version = "1.0"

    def __init__(self, destroy_operators: Sequence[str], repair_operators: Sequence[str]):
        destroy_list = list(destroy_operators)
        repair_list = list(repair_operators)
        if not destroy_list or not repair_list:
            raise ValueError("Operator lists must be non-empty")

        self.destroy_operators = destroy_list
        self.repair_operators = repair_list
        self._num_destroy = len(destroy_list)
        self._num_repair = len(repair_list)
        self._space = spaces.Discrete(self._num_destroy * self._num_repair)

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> tuple[int, int]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")
        destroy_idx = action_idx // self._num_repair
        repair_idx = action_idx % self._num_repair
        return destroy_idx, repair_idx

    def action_to_id(self, destroy_idx: int, repair_idx: int) -> int:
        if destroy_idx < 0 or destroy_idx >= self._num_destroy:
            raise ValueError(f"Destroy index {destroy_idx} out of range")
        if repair_idx < 0 or repair_idx >= self._num_repair:
            raise ValueError(f"Repair index {repair_idx} out of range")
        return destroy_idx * self._num_repair + repair_idx

    def sample(self) -> int:
        return int(self._space.sample())
