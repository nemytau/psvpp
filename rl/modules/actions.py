"""Action space implementations for the ALNS RL environment.

Experiment -> ID -> action module mapping (used for run-tag / folder naming):

- Experiment 1:
    - ID 1: `op_pair_v1` -> folder tag `exp1_id1_op_pair_v1`
    - ID 2: `op_pair_mapped_v1` -> folder tag `exp1_id2_op_pair_mapped_v1`
    - ID 5: `destroy_repair_many_v1` -> folder tag `exp1_id5_destroy_repair_many_v1`
- Experiment 2:
    - ID 3: `improvement_chain_v1` -> folder tag `exp2_id3_improvement_chain_v1`
    - ID 4: `improvement_many_v1` -> folder tag `exp2_id4_improvement_many_v1`

Models within the same experiment should be grouped under the same `expX_...` prefix
and include the `idY` identifier so comparisons and archives are consistent.
"""

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


@register_action("destroy_repair_many_v1")
class DestroyRepairManyActionSpace(BaseActionSpace):
    """Discrete actions over a large destroy/repair grid with no improvement phase.

    This action space is meant for the experiment that compares many parameterized
    destroy and repair operators against basic ALNS without an improvement step.
    It mirrors the Rust destroy/repair families and keeps the mapping stable as the
    operator grid grows.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        del improvement_operators

        destroy_list = list(destroy_operators)
        repair_list = list(repair_operators)
        if not destroy_list or not repair_list:
            raise ValueError("Operator lists must be non-empty")

        self.destroy_operators = destroy_list
        self.repair_operators = repair_list
        self._num_destroy = len(destroy_list)
        self._num_repair = len(repair_list)
        self._pairs = [
            (d_idx, r_idx)
            for d_idx in range(self._num_destroy)
            for r_idx in range(self._num_repair)
        ]
        self._pair_index = {pair: idx for idx, pair in enumerate(self._pairs)}
        self._space = spaces.Discrete(len(self._pairs))

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")
        destroy_idx, repair_idx = self._pairs[action_idx]
        return destroy_idx, repair_idx, None

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        del improvement_idx
        if destroy_idx is None or repair_idx is None:
            raise ValueError("Destroy and repair indices required for destroy_repair_many_v1")
        if destroy_idx < 0 or destroy_idx >= self._num_destroy:
            raise ValueError(f"Destroy index {destroy_idx} out of range")
        if repair_idx < 0 or repair_idx >= self._num_repair:
            raise ValueError(f"Repair index {repair_idx} out of range")

        pair_idx = self._pair_index.get((destroy_idx, repair_idx))
        if pair_idx is None:
            raise ValueError(f"Destroy/repair pair ({destroy_idx}, {repair_idx}) not registered")
        return pair_idx

    def sample(self) -> int:
        return int(self._space.sample())

    def num_destroy(self) -> int:
        return self._num_destroy

    def num_repair(self) -> int:
        return self._num_repair

    def num_pairs(self) -> int:
        return len(self._pairs)


@register_action("improvement_only_v1")
class ImprovementOnlyActionSpace(BaseActionSpace):
    """Discrete actions for improvement selection on top of ALNS backbone.

    Action mapping:
    - 0: skip improvement (run destroy/repair backbone only)
    - 1..N: apply one selected improvement operator

    The class filters improvement operators to a preferred subset while preserving
    their engine indices. Preferred names: voyage_number_reduction,
    fleet_and_cost_reduction, deep_relocation, deep_swap.
    """

    version = "1.0"

    _PREFERRED_IMPROVEMENTS = tuple(
        f"{base}_v{i}"
        for base in ("voyage_number_reduction", "fleet_and_cost_reduction", "deep_relocation", "deep_swap")
        for i in range(1, 11)
    )

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        del destroy_operators, repair_operators

        all_improvements = list(improvement_operators or [])
        name_to_idx = {name: idx for idx, name in enumerate(all_improvements)}

        selected_pairs = [
            (name, name_to_idx[name])
            for name in self._PREFERRED_IMPROVEMENTS
            if name in name_to_idx
        ]

        if not selected_pairs and all_improvements:
            selected_pairs = [(name, idx) for idx, name in enumerate(all_improvements)]

        self.improvement_operators = [name for name, _ in selected_pairs]
        self._improvement_indices = [idx for _, idx in selected_pairs]
        self._improvement_to_action = {
            idx: action_id + 1
            for action_id, idx in enumerate(self._improvement_indices)
        }

        self._space = spaces.Discrete(1 + len(self._improvement_indices))

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")

        if action_idx == 0:
            return None, None, None

        mapped_idx = action_idx - 1
        if mapped_idx < 0 or mapped_idx >= len(self._improvement_indices):
            raise ValueError(f"Improvement action index {action_idx} out of range")

        return None, None, self._improvement_indices[mapped_idx]

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if destroy_idx not in {None, -1} or repair_idx not in {None, -1}:
            raise ValueError("Improvement-only action space does not accept destroy/repair indices")

        if improvement_idx is None:
            return 0

        action_id = self._improvement_to_action.get(int(improvement_idx))
        if action_id is None:
            raise ValueError(
                f"Improvement index {improvement_idx} not available in improvement_only_v1 action space"
            )
        return action_id

    def sample(self) -> int:
        return int(self._space.sample())

    def num_improvement(self) -> int:
        return len(self._improvement_indices)


@register_action("improvement_chain_v1")
class ImprovementChainActionSpace(BaseActionSpace):
    """Action space that allows learned improvement chains.

    Action mapping:
    - 0: run backbone destroy+repair step (no explicit improvement)
    - 1..N: run one improvement-only step with selected operator

    Repeated 1..N actions naturally form an improvement chain before the next
    backbone action is chosen.
    """

    version = "1.0"

    _PREFERRED_IMPROVEMENTS = tuple(
        f"{base}_v{i}"
        for base in ("voyage_number_reduction", "fleet_and_cost_reduction", "deep_relocation", "deep_swap")
        for i in range(1, 11)
    )

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        del destroy_operators, repair_operators

        all_improvements = list(improvement_operators or [])
        name_to_idx = {name: idx for idx, name in enumerate(all_improvements)}

        selected_pairs = [
            (name, name_to_idx[name])
            for name in self._PREFERRED_IMPROVEMENTS
            if name in name_to_idx
        ]

        if not selected_pairs and all_improvements:
            selected_pairs = [(name, idx) for idx, name in enumerate(all_improvements)]

        self.improvement_operators = [name for name, _ in selected_pairs]
        self._improvement_indices = [idx for _, idx in selected_pairs]
        self._improvement_to_action = {
            idx: action_id + 1
            for action_id, idx in enumerate(self._improvement_indices)
        }

        self._space = spaces.Discrete(1 + len(self._improvement_indices))

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")

        if action_idx == 0:
            return None, None, None

        mapped_idx = action_idx - 1
        if mapped_idx < 0 or mapped_idx >= len(self._improvement_indices):
            raise ValueError(f"Improvement action index {action_idx} out of range")

        return None, None, self._improvement_indices[mapped_idx]

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if destroy_idx not in {None, -1} or repair_idx not in {None, -1}:
            raise ValueError("improvement_chain_v1 does not accept explicit destroy/repair indices")

        if improvement_idx is None:
            return 0

        action_id = self._improvement_to_action.get(int(improvement_idx))
        if action_id is None:
            raise ValueError(
                f"Improvement index {improvement_idx} not available in improvement_chain_v1"
            )
        return action_id

    def sample(self) -> int:
        return int(self._space.sample())

    def num_improvement(self) -> int:
        return len(self._improvement_indices)


@register_action("improvement_many_v1")
class ImprovementManyActionSpace(BaseActionSpace):
    """Discrete actions selecting among parameterized improvement variants.

    Action mapping:
    - 0: skip improvement (run backbone destroy+repair)
    - 1..M: apply one selected improvement variant (mapped to engine indices)
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        del destroy_operators, repair_operators

        all_improvements = list(improvement_operators or [])
        name_to_idx = {name: idx for idx, name in enumerate(all_improvements)}

        base_names = (
            "voyage_number_reduction",
            "fleet_and_cost_reduction",
            "deep_relocation",
            "deep_swap",
        )

        selected_pairs: list[tuple[str, int]] = []
        for base in base_names:
            for i in range(1, 11):
                name = f"{base}_v{i}"
                if name in name_to_idx:
                    selected_pairs.append((name, name_to_idx[name]))

        if not selected_pairs and all_improvements:
            selected_pairs = [(name, idx) for idx, name in enumerate(all_improvements)]

        self.improvement_operators = [name for name, _ in selected_pairs]
        self._improvement_indices = [idx for _, idx in selected_pairs]
        self._improvement_to_action = {
            idx: action_id + 1 for action_id, idx in enumerate(self._improvement_indices)
        }

        self._space = spaces.Discrete(1 + len(self._improvement_indices))

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")

        if action_idx == 0:
            return None, None, None

        mapped_idx = action_idx - 1
        if mapped_idx < 0 or mapped_idx >= len(self._improvement_indices):
            raise ValueError(f"Improvement action index {action_idx} out of range")

        return None, None, self._improvement_indices[mapped_idx]

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if destroy_idx not in {None, -1} or repair_idx not in {None, -1}:
            raise ValueError("improvement_many_v1 does not accept explicit destroy/repair indices")

        if improvement_idx is None:
            return 0

        action_id = self._improvement_to_action.get(int(improvement_idx))
        if action_id is None:
            raise ValueError(
                f"Improvement index {improvement_idx} not available in improvement_many_v1"
            )
        return action_id

    def sample(self) -> int:
        return int(self._space.sample())

    def num_improvement(self) -> int:
        return len(self._improvement_indices)

@register_action("op_pair_mapped_v1")
class OperatorPairMappedActionSpace(BaseActionSpace):
    """Wrapper around `OperatorPairActionSpace` that selects a small set of
    parameterized operators from the engine-provided lists which are analogical
    to the single, non-parameterized operators used previously.

    The wrapper picks representative variants (middle-of-range) for each
    destroy family and chooses repair variants matching previous defaults
    (e.g. `deep_greedy_rank_1`, `k_regret_2_s3`, `k_regret_3_s3`) when
    available. It returns indices relative to the engine operator lists so the
    rest of the environment and logger remain consistent.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        original_destroy = list(destroy_operators)
        original_repair = list(repair_operators)

        def pick_mid(candidates: list[str]) -> Optional[str]:
            if not candidates:
                return None
            return candidates[len(candidates) // 2]

        # Select representative destroy operators by family
        shaw = [n for n in original_destroy if n.startswith("shaw_removal_v")]
        random_visit = [n for n in original_destroy if n.startswith("random_visit_removal_v")]
        worst_visit = [n for n in original_destroy if n.startswith("worst_visit_removal_v")]

        selected_destroy_names: list[str] = []
        for group in (shaw, random_visit, worst_visit):
            v = pick_mid(group)
            if v is not None:
                selected_destroy_names.append(v)

        # Select representative repair operators
        selected_repair_names: list[str] = []
        # prefer deep_greedy_rank_1 when available (matches earlier candidate_rank=1)
        if any(n.startswith("deep_greedy_rank_1") for n in original_repair):
            selected_repair_names.append(next(n for n in original_repair if n.startswith("deep_greedy_rank_1")))
        else:
            dg = [n for n in original_repair if n.startswith("deep_greedy_rank_")]
            if dg:
                selected_repair_names.append(dg[0])

        # prefer k_regret_2_s3 / k_regret_3_s3 when present
        for pref in ("k_regret_2_s3", "k_regret_3_s3"):
            if pref in original_repair:
                selected_repair_names.append(pref)

        # fallback to middle k_regret variants if specific s3 variants missing
        if len(selected_repair_names) < 3:
            k2 = [n for n in original_repair if n.startswith("k_regret_2_")]
            k3 = [n for n in original_repair if n.startswith("k_regret_3_")]
            if k2:
                v = pick_mid(k2)
                if v:
                    selected_repair_names.append(v)
            if k3:
                v = pick_mid(k3)
                if v:
                    selected_repair_names.append(v)

        # Ensure uniqueness and presence in original lists
        selected_destroy_names = [n for i, n in enumerate(selected_destroy_names) if n in original_destroy]
        selected_repair_names = [n for i, n in enumerate(selected_repair_names) if n in original_repair]

        def _pick_evenly(items: list[str], count: int) -> list[str]:
            if not items:
                return []
            count = min(count, len(items))
            if count <= 0:
                return []
            if count == 1:
                return [items[len(items) // 2]]
            step = max(1, len(items) // count)
            picks: list[str] = []
            idx = 0
            while len(picks) < count and idx < len(items):
                picks.append(items[idx])
                idx += step
            # Ensure we have exactly `count` elements by appending from the end if needed
            i = 1
            while len(picks) < count and i <= len(items):
                candidate = items[-i]
                if candidate not in picks:
                    picks.append(candidate)
                i += 1
            return picks

        # If preferences failed to match any operators, fall back to evenly spaced
        # representatives from the engine-provided lists. Only raise if the engine
        # supplied no operators at all.
        if not original_destroy:
            raise ValueError("Engine provided no destroy operators for op_pair_mapped_v1")
        if not original_repair:
            raise ValueError("Engine provided no repair operators for op_pair_mapped_v1")

        if not selected_destroy_names:
            # pick up to three representative destroys
            selected_destroy_names = _pick_evenly(original_destroy, 3)
        if not selected_repair_names:
            # pick up to three representative repairs
            selected_repair_names = _pick_evenly(original_repair, 3)

        # Map selected names back to engine indices
        self._destroy_name_to_engine_idx = [original_destroy.index(n) for n in selected_destroy_names]
        self._repair_name_to_engine_idx = [original_repair.index(n) for n in selected_repair_names]

        # Build an inner OperatorPairActionSpace over the selected names so the
        # RL agent sees a compact, human-readable action mapping.
        self._inner = OperatorPairActionSpace(
            destroy_operators=selected_destroy_names,
            repair_operators=selected_repair_names,
            improvement_operators=list(improvement_operators or []),
        )

    def gym_space(self) -> spaces.Discrete:
        return self._inner.gym_space()

    def n(self) -> int:
        return int(self._inner.n())

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        d_idx_local, r_idx_local, imp = self._inner.id_to_action(action_idx)
        if d_idx_local is None or r_idx_local is None:
            return None, None, imp
        # Map local indices back to engine-global indices
        return self._destroy_name_to_engine_idx[int(d_idx_local)], self._repair_name_to_engine_idx[int(r_idx_local)], None

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        # Accept engine-global destroy/repair indices and map them into the
        # compact inner space before delegating to the inner action_to_id.
        if improvement_idx is not None:
            # delegate improvement-only semantics to inner
            return self._inner.action_to_id(None, None, improvement_idx)

        if destroy_idx is None or repair_idx is None:
            raise ValueError("Destroy and repair indices required for op_pair_mapped_v1")

        try:
            local_d = self._destroy_name_to_engine_idx.index(int(destroy_idx))
            local_r = self._repair_name_to_engine_idx.index(int(repair_idx))
        except ValueError as exc:
            raise ValueError("Requested engine operator indices are not part of the mapped selection") from exc

        return self._inner.action_to_id(local_d, local_r, None)

    def sample(self) -> int:
        return int(self._inner.sample())
