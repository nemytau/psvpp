"""Action space implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from gymnasium import spaces

from rl.registries import register_action


class BaseActionSpace:
    """Base class for action-space adapters."""

    version: str = "base"

    def gym_space(self) -> spaces.Space:  # pragma: no cover - interface only
        raise NotImplementedError

    def n(self) -> int:  # pragma: no cover - interface only
        raise NotImplementedError

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:  # pragma: no cover - interface only
        raise NotImplementedError

    def action_to_id(
        self,
        destroy_idx: int,
        repair_idx: int,
        improvement_idx: Optional[int] = None,
    ) -> int:  # pragma: no cover - interface only
        raise NotImplementedError

    def sample(self) -> int:  # pragma: no cover - interface only
        raise NotImplementedError


@register_action("op_pair_v1")
class OperatorPairActionSpace(BaseActionSpace):
    """Discrete actions covering destroy/repair pairs and independent improvements.

    The first ``num_destroy * num_repair`` actions map to destroy/repair pairs with no
    improvement operator selected. The remaining ``num_improvement`` actions represent
    improvement-only steps where no destroy/repair operator is chosen.
    """

    version = "1.3"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        destroy_list = list(destroy_operators)
        repair_list = list(repair_operators)
        improvement_list = list(improvement_operators or [])
        if not destroy_list or not repair_list:
            raise ValueError("Operator lists must be non-empty")

        self.destroy_operators = destroy_list
        self.repair_operators = repair_list
        self.improvement_operators = improvement_list
        self._num_destroy = len(destroy_list)
        self._num_repair = len(repair_list)
        self._num_improvement = len(improvement_list)

        self._pairs = [
            (d_idx, r_idx)
            for d_idx in range(self._num_destroy)
            for r_idx in range(self._num_repair)
        ]
        self._num_pairs = len(self._pairs)
        self._pair_index = {pair: idx for idx, pair in enumerate(self._pairs)}

        total_actions = self._num_pairs + self._num_improvement
        self._space = spaces.Discrete(total_actions)

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")

        if action_idx < self._num_pairs:
            destroy_idx, repair_idx = self._pairs[action_idx]
            return destroy_idx, repair_idx, None

        improvement_offset = action_idx - self._num_pairs
        if improvement_offset < 0 or improvement_offset >= self._num_improvement:
            raise ValueError(
                f"Improvement action index {action_idx} out of bounds for {self._num_improvement} improvement operators"
            )

        return None, None, improvement_offset

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if improvement_idx is None:
            if destroy_idx is None or repair_idx is None:
                raise ValueError("Destroy and repair indices required for pair actions")
            if destroy_idx < 0 or destroy_idx >= self._num_destroy:
                raise ValueError(f"Destroy index {destroy_idx} out of range")
            if repair_idx < 0 or repair_idx >= self._num_repair:
                raise ValueError(f"Repair index {repair_idx} out of range")

            pair_idx = self._pair_index.get((destroy_idx, repair_idx))
            if pair_idx is None:
                raise ValueError(
                    f"Destroy/repair pair ({destroy_idx}, {repair_idx}) not registered"
                )
            return pair_idx

        if self._num_improvement <= 0:
            raise ValueError("Improvement action requested but no improvement operators registered")
        if improvement_idx < 0 or improvement_idx >= self._num_improvement:
            raise ValueError(f"Improvement index {improvement_idx} out of range")
        if destroy_idx not in {None, -1} or repair_idx not in {None, -1}:
            raise ValueError(
                "Improvement-only actions must omit destroy and repair indices"
            )

        return self._num_pairs + improvement_idx

    def sample(self) -> int:
        return int(self._space.sample())

    def num_pairs(self) -> int:
        return self._num_pairs

    def num_improvement(self) -> int:
        return self._num_improvement

    def first_improvement_action(self) -> Optional[int]:
        if self._num_improvement <= 0:
            return None
        return self._num_pairs
