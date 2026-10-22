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


class OperatorPairActionSpace(BaseActionSpace):
    """Discrete actions as a full D×R×I product.

    Each action selects one destroy operator, one repair operator, and one
    improvement operator simultaneously.  Action index is encoded as:

        action_idx = d_local * (n_repair * n_improvement)
                   + r_local * n_improvement
                   + i_local

    All three operators are required; pass filtered lists to this class.
    """

    version = "2.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        destroy_list = list(destroy_operators)
        repair_list = list(repair_operators)
        improvement_list = list(improvement_operators or [])
        if not destroy_list or not repair_list or not improvement_list:
            raise ValueError("All three operator lists must be non-empty for D×R×I action space")

        self.destroy_operators = destroy_list
        self.repair_operators = repair_list
        self.improvement_operators = improvement_list
        self._num_destroy = len(destroy_list)
        self._num_repair = len(repair_list)
        self._num_improvement = len(improvement_list)

        self._space = spaces.Discrete(self._num_destroy * self._num_repair * self._num_improvement)

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")
        d_local = action_idx // (self._num_repair * self._num_improvement)
        r_local = (action_idx // self._num_improvement) % self._num_repair
        i_local = action_idx % self._num_improvement
        return d_local, r_local, i_local

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if destroy_idx is None or repair_idx is None or improvement_idx is None:
            raise ValueError("D×R×I action space requires all three indices")
        return (int(destroy_idx) * self._num_repair * self._num_improvement
                + int(repair_idx) * self._num_improvement
                + int(improvement_idx))

    def sample(self) -> int:
        return int(self._space.sample())

    def num_pairs(self) -> int:
        return self._num_destroy * self._num_repair

    def num_improvement(self) -> int:
        return self._num_improvement


@register_action("adaptive_weighted_v1")
class AdaptiveWeightedActionSpace(BaseActionSpace):
    """Single-action space that always triggers the adaptive weighted engine,
    restricted to the same 3x3 destroy/repair pool as op_pair_v1 (see
    _base_destroy_repair) -- matching Kisialiou's original paper's operator
    selection, and ensuring baseline/Kisialiou/RL are all comparing the same
    operator set.

    Action 0 → (None, None, None): the engine selects destroy and repair via
    weighted roulette (adaptive ALNS) over the restricted 3x3 pool, updating
    weights every segment. Used as the default baseline so it runs true
    adaptive ALNS, not random explicit pair selection.

    CAUTION: this class previously stored the *unrestricted* full operator
    catalog (a bug -- restrict_operators() in rl_alns_environment.py became a
    no-op as a result). If you need that old unrestricted-pool behavior
    (e.g. to reproduce a specific historical run), use adaptive_weighted_full_v1
    instead -- do not "fix" this class back to unrestricted. (Formerly named
    backbone_v1 / BackboneActionSpace -- renamed 2026-08-12 for clarity; old
    run manifests may still reference "backbone_v1" by the pre-rename name.)
    """

    version = "2.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        selected_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)
        self.destroy_operators = selected_destroy
        self.repair_operators = selected_repair
        self.improvement_operators: list = []
        self._space = spaces.Discrete(1)

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return 1

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        return None, None, None

    def action_to_id(self, destroy_idx, repair_idx, improvement_idx=None) -> int:
        return 0

    def sample(self) -> int:
        return 0

    def num_pairs(self) -> int:
        return 1

    def num_improvement(self) -> int:
        return 0


@register_action("adaptive_weighted_v2")
class AdaptiveWeightedV2ActionSpace(AdaptiveWeightedActionSpace):
    """adaptive_weighted_v1 plus `first_fit_insertion_v1` as a 4th repair operator.

    Baseline/Kisialiou operator pool of the 2026-10 rerun of Experiment 8: the RL action
    spaces `op_pair_v2` / `op_triple_v2` / `improvement_chain_v2` offer first-fit insertion,
    so the non-learned methods get it too (pool = 3 Kisialiou-calibrated destroy operators x
    4 repair operators). Pass this as --baseline-action-module: the same pool restricts the
    adaptive baseline and Kisialiou's ALNS (run_kisialiou_episode takes the baseline module).
    adaptive_weighted_v1 stays unchanged so earlier caches remain reproducible.
    """

    version = "2.1"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        super().__init__(destroy_operators, repair_operators, improvement_operators, size_bucket=size_bucket)
        first_fit = _first(list(repair_operators), "first_fit_insertion_v1")
        if first_fit is not None and first_fit not in self.repair_operators:
            self.repair_operators = [*self.repair_operators, first_fit]


@register_action("adaptive_weighted_full_v1")
class AdaptiveWeightedFullActionSpace(BaseActionSpace):
    """Single-action space that triggers the adaptive weighted engine over
    the FULL registered operator catalog (every versioned variant), not the
    restricted 3x3 pool. This is what "adaptive_weighted_v1" (formerly
    backbone_v1) incorrectly did before the pool-restriction bug was found
    and fixed (2026-08-07) -- kept here, honestly named, as a deliberate
    "adaptive search over the whole catalog" option, distinct from the
    canonical restricted baseline (adaptive_weighted_v1).

    Action 0 → (None, None, None): destroy/repair chosen via weighted
    roulette over ALL registered operators (no restriction). (Formerly named
    backbone_full_v1 / BackboneFullActionSpace -- renamed 2026-08-12.)
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
    ) -> None:
        self.destroy_operators = list(destroy_operators)
        self.repair_operators = list(repair_operators)
        self.improvement_operators: list = []
        self._space = spaces.Discrete(1)

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return 1

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        return None, None, None

    def action_to_id(self, destroy_idx, repair_idx, improvement_idx=None) -> int:
        return 0

    def sample(self) -> int:
        return 0

    def num_pairs(self) -> int:
        return 1

    def num_improvement(self) -> int:
        return 0


def _first(names: Sequence[str], prefix: str) -> Optional[str]:
    """Return the first name starting with `prefix`, or None."""
    return next((n for n in names if n.startswith(prefix)), None)


def _base_destroy_repair(
    d_all: list, r_all: list, size_bucket: str = "large"
) -> Tuple[list, list]:
    """Pick the Kisialiou-paper-calibrated destroy operators and the standard
    repair set.

    size_bucket selects which xi (removal-fraction) calibration to use, per
    Kisialiou et al. (2018) Table 1 / Section 3.3: "small" for instances with
    <15 installations (xi in [0.18,0.23]), "large" for >=15 (xi in
    [0.13,0.18]). Our small/medium/large test sizes have 10/20/40
    installations respectively, so "small" maps only to our "small" size and
    "large" covers both "medium" and "large".
    """
    bucket = size_bucket if size_bucket in ("small", "large") else "large"
    selected_destroy = [n for n in (
        _first(d_all, f"shaw_removal_kisialiou_{bucket}_v1"),
        _first(d_all, f"random_visit_removal_kisialiou_{bucket}_v1"),
        _first(d_all, f"worst_visit_removal_kisialiou_{bucket}_v1"),
    ) if n is not None]
    selected_repair = [n for n in (
        _first(r_all, "deep_greedy_rank_1"),
        next((x for x in r_all if x == "k_regret_2" or x.startswith("k_regret_2_")), None),
        next((x for x in r_all if x == "k_regret_3" or x.startswith("k_regret_3_")), None),
    ) if n is not None]
    return selected_destroy, selected_repair


_BASE_IMPROVEMENTS = (
    "voyage_number_reduction_v1",
    "fleet_and_cost_reduction_kisialiou_v1",  # lambda_max=35, Kisialiou et al. (2018) Table 1
    "deep_relocation_v10",           # unlimited, run to convergence
    "deep_swap_v10",                  # unlimited, run to convergence
)


@register_action("op_pair_v1")
class OpPairV1ActionSpace(BaseActionSpace):
    """RL selects destroy+repair pairs only — no improvement phase.

    Intended as exp0 baseline: compare pure RL pair selection against
    random-pair ALNS without any improvement operator.

    Action space: 3 destroy × 3 repair = 9 actions.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        selected_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)

        self.destroy_operators = selected_destroy
        self.repair_operators = selected_repair
        self.improvement_operators: list = []
        self._num_destroy = len(selected_destroy)
        self._num_repair = len(selected_repair)
        self._pairs = [(d, r) for d in range(self._num_destroy) for r in range(self._num_repair)]
        self._space = spaces.Discrete(len(self._pairs))
        # Engine-index mappings
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")
        d_local, r_local = self._pairs[action_idx]
        return self._d_engine[d_local], self._r_engine[r_local], None

    def action_to_id(self, destroy_idx, repair_idx, improvement_idx=None) -> int:
        local_d = self._d_engine.index(int(destroy_idx))
        local_r = self._r_engine.index(int(repair_idx))
        return local_d * self._num_repair + local_r

    def sample(self) -> int:
        return int(self._space.sample())

    def num_pairs(self) -> int:
        return len(self._pairs)

    def num_improvement(self) -> int:
        return 0


@register_action("op_pair_v2")
class OpPairV2ActionSpace(BaseActionSpace):
    """Same as op_pair_v1, plus `first_fit_insertion_v1` as a 4th repair choice.

    Flat destroy x repair control (no improvement phase at all) -- isolates
    whether first_fit_insertion_v1's effect on training dynamics depends on
    the larger D x R x I action space (op_triple_v2), or shows up already in
    the simplest possible action space.

    Action space: 3 destroy x 4 repair = 12 actions.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        selected_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)
        first_fit = _first(r_all, "first_fit_insertion_v1")
        if first_fit is not None:
            selected_repair = [*selected_repair, first_fit]

        self.destroy_operators = selected_destroy
        self.repair_operators = selected_repair
        self.improvement_operators: list = []
        self._num_destroy = len(selected_destroy)
        self._num_repair = len(selected_repair)
        self._pairs = [(d, r) for d in range(self._num_destroy) for r in range(self._num_repair)]
        self._space = spaces.Discrete(len(self._pairs))
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]

    def gym_space(self) -> spaces.Discrete:
        return self._space

    def n(self) -> int:
        return int(self._space.n)

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not self._space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} out of range [0, {self._space.n})")
        d_local, r_local = self._pairs[action_idx]
        return self._d_engine[d_local], self._r_engine[r_local], None

    def action_to_id(self, destroy_idx, repair_idx, improvement_idx=None) -> int:
        local_d = self._d_engine.index(int(destroy_idx))
        local_r = self._r_engine.index(int(repair_idx))
        return local_d * self._num_repair + local_r

    def sample(self) -> int:
        return int(self._space.sample())

    def num_pairs(self) -> int:
        return len(self._pairs)

    def num_improvement(self) -> int:
        return 0


@register_action("op_triple_v1")
class OpTripleV1ActionSpace(OperatorPairActionSpace):
    """D×R×I action space using one base (v1) operator per family.

    Action space: 3 destroy × 3 repair × 4 improvement = 36 actions.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        i_all = list(improvement_operators or [])
        selected_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)
        selected_improve = [n for n in _BASE_IMPROVEMENTS if n in i_all]

        super().__init__(
            destroy_operators=selected_destroy,
            repair_operators=selected_repair,
            improvement_operators=selected_improve,
        )
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]
        self._i_engine = [i_all.index(n) for n in selected_improve]

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        d_local, r_local, i_local = super().id_to_action(action_idx)
        return self._d_engine[d_local], self._r_engine[r_local], self._i_engine[i_local]


@register_action("op_triple_v2")
class OpTripleV2ActionSpace(OperatorPairActionSpace):
    """Same as op_triple_v1, plus `first_fit_insertion_v1` as a 4th repair choice.

    Destroy and improvement operator selection is identical to op_triple_v1
    (Kisialiou-calibrated destroy triplet, same 4-operator improvement set).
    Adds the cheap, non-cost-ranked FirstFitInsertion repair operator (see
    rust_alns/src/operators/repair/first_fit_insertion.rs) to the repair pool,
    to test whether RL benefits from having a fast fallback repair available
    alongside the cost-ranked DeepGreedyInsertion/KRegretInsertion operators.

    Not used by baseline or Kisialiou -- those stay on op_pair_v1/
    adaptive_weighted_v1's original 3-operator repair pool, deliberately
    unchanged, so they remain a fixed reference point.

    Action space: 3 destroy x 4 repair x 4 improvement = 48 actions.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        i_all = list(improvement_operators or [])
        selected_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)
        first_fit = _first(r_all, "first_fit_insertion_v1")
        if first_fit is not None:
            selected_repair = [*selected_repair, first_fit]
        selected_improve = [n for n in _BASE_IMPROVEMENTS if n in i_all]

        super().__init__(
            destroy_operators=selected_destroy,
            repair_operators=selected_repair,
            improvement_operators=selected_improve,
        )
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]
        self._i_engine = [i_all.index(n) for n in selected_improve]

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        d_local, r_local, i_local = super().id_to_action(action_idx)
        return self._d_engine[d_local], self._r_engine[r_local], self._i_engine[i_local]


# Parameter levels for the "many operators" action space (NORM 2026): the same operator
# families as op_triple_v2, each offered at several parameter settings so RL also chooses
# *how hard* to destroy / improve, not just which family.
_MANY_IMPROVEMENTS = (
    "voyage_number_reduction_v1",             # search_k = 1
    "voyage_number_reduction_v3",             # search_k = 3
    "voyage_number_reduction_v5",             # search_k = 5
    "fleet_and_cost_reduction_v1",            # lambda_max = 8
    "fleet_and_cost_reduction_kisialiou_v1",  # lambda_max = 35 (Kisialiou calibration)
    "fleet_and_cost_reduction_v10",           # lambda_max = 80
    "deep_relocation_v1",                     # max_moves = 1
    "deep_relocation_m10_v1",                 # max_moves = 10
    "deep_relocation_v10",                    # unlimited
    "deep_swap_v1",                           # max_moves = 1
    "deep_swap_m10_v1",                       # max_moves = 10
    "deep_swap_v10",                          # unlimited
)


@register_action("op_triple_many_v1")
class OpTripleManyV1ActionSpace(OperatorPairActionSpace):
    """op_triple_v2 with parameterised operator variants (NORM 2026 action-space ablation).

    Destroy: Shaw / random / worst removal, each at the Kisialiou-calibrated removal ratio
    (size-bucket dependent) and at a high ratio (xi in [0.30, 0.40]); all other destroy
    parameters stay at the Kisialiou calibration, so only the ratio differs.
    Repair: unchanged from op_triple_v2 (greedy, regret-2, regret-3, first-fit).
    Improvement: 3 intensity levels per family (see _MANY_IMPROVEMENTS).

    Action space: 6 destroy x 4 repair x 12 improvement = 288 actions.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        i_all = list(improvement_operators or [])
        calibrated_destroy, selected_repair = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)
        high_destroy = [
            "shaw_removal_high_v1",
            "random_visit_removal_high_v1",
            "worst_visit_removal_high_v1",
        ]
        missing = [n for n in high_destroy + list(_MANY_IMPROVEMENTS) if n not in d_all + i_all]
        if missing:
            raise ValueError(f"op_triple_many_v1 needs operators missing from the engine (rebuild rust_alns?): {missing}")
        selected_destroy = [op for pair in zip(calibrated_destroy, high_destroy) for op in pair]
        first_fit = _first(r_all, "first_fit_insertion_v1")
        if first_fit is not None:
            selected_repair = [*selected_repair, first_fit]
        selected_improve = list(_MANY_IMPROVEMENTS)

        super().__init__(
            destroy_operators=selected_destroy,
            repair_operators=selected_repair,
            improvement_operators=selected_improve,
        )
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]
        self._i_engine = [i_all.index(n) for n in selected_improve]

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        d_local, r_local, i_local = super().id_to_action(action_idx)
        return self._d_engine[d_local], self._r_engine[r_local], self._i_engine[i_local]


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
    - 1..4: run one improvement-only step with one of the 4 base operators

    Uses only the 4 base improvement operators (one per family).
    Repeated 1..4 actions form an improvement chain before the next backbone step.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        # Store the base 3+3 for engine operator restriction
        self.destroy_operators, self.repair_operators = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)

        all_improvements = list(improvement_operators or [])
        name_to_idx = {name: idx for idx, name in enumerate(all_improvements)}

        # Use only the 4 base improvement operators — one per family
        selected_pairs = [
            (name, name_to_idx[name])
            for name in _BASE_IMPROVEMENTS
            if name in name_to_idx
        ]

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


@register_action("improvement_chain_v2")
class ImprovementChainV2ActionSpace(ImprovementChainActionSpace):
    """Same as improvement_chain_v1, plus `first_fit_insertion_v1` in the backbone's repair pool.

    RL's action space is unchanged (still 1 backbone + 4 improvement-operator
    choices) -- the backbone destroy/repair pair is always resolved by the
    engine's own adaptive-weighted roulette (mode="weighted"), never chosen
    explicitly by RL, for both v1 and v2. What changes is which repair
    operators that roulette is allowed to pick from: v1 restricts to the 3
    standard cost-ranked operators (deep_greedy_rank_1, k_regret_2,
    k_regret_3); v2 adds first_fit_insertion_v1 as a 4th, cheap option.

    Motivation: the v1 small-eval result (2026-08-24) showed real destroy/
    repair diversity (the weighted roulette works fine) but a much lower
    restart count within the same wall-time-matched eval budget than
    op_triple_v2 achieved (1,241 vs 2,215) -- cost-ranked-only repair is
    slower per iteration, so fewer restarts fit. This tests whether giving
    the backbone the same cheap escape hatch op_triple_v2 had closes that
    gap without needing RL to choose the pair at all.
    """

    version = "1.0"

    def __init__(
        self,
        destroy_operators: Sequence[str],
        repair_operators: Sequence[str],
        improvement_operators: Optional[Sequence[str]] = None,
        size_bucket: str = "large",
    ) -> None:
        super().__init__(
            destroy_operators=destroy_operators,
            repair_operators=repair_operators,
            improvement_operators=improvement_operators,
            size_bucket=size_bucket,
        )
        first_fit = _first(list(repair_operators), "first_fit_insertion_v1")
        if first_fit is not None and first_fit not in self.repair_operators:
            self.repair_operators = [*self.repair_operators, first_fit]


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
        size_bucket: str = "large",
    ) -> None:
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        self.destroy_operators, self.repair_operators = _base_destroy_repair(d_all, r_all, size_bucket=size_bucket)

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
        d_all = list(destroy_operators)
        r_all = list(repair_operators)
        i_all = list(improvement_operators or [])

        def pick_mid(items: list[str]) -> Optional[str]:
            return items[len(items) // 2] if items else None

        # One representative destroy per family — mid-range parameterised version
        selected_destroy = [n for n in (
            pick_mid([x for x in d_all if x.startswith("shaw_removal_v")]),
            pick_mid([x for x in d_all if x.startswith("random_visit_removal_v")]),
            pick_mid([x for x in d_all if x.startswith("worst_visit_removal_v")]),
        ) if n is not None]

        # Repair: greedy rank_1, k_regret_2, k_regret_3 (exact or first match)
        selected_repair = [n for n in (
            _first(r_all, "deep_greedy_rank_1"),
            next((x for x in r_all if x == "k_regret_2" or x.startswith("k_regret_2_")), None),
            next((x for x in r_all if x == "k_regret_3" or x.startswith("k_regret_3_")), None),
        ) if n is not None]

        # One representative improvement per family — mid-range parameterised version
        selected_improve = [n for n in (
            pick_mid([x for x in i_all if x.startswith("voyage_number_reduction_v")]),
            pick_mid([x for x in i_all if x.startswith("fleet_and_cost_reduction_v")]),
            pick_mid([x for x in i_all if x.startswith("deep_relocation_v")]),
            pick_mid([x for x in i_all if x.startswith("deep_swap_v")]),
        ) if n is not None]

        # Store engine-index mappings
        self._d_engine = [d_all.index(n) for n in selected_destroy]
        self._r_engine = [r_all.index(n) for n in selected_repair]
        self._i_engine = [i_all.index(n) for n in selected_improve]

        self._inner = OperatorPairActionSpace(
            destroy_operators=selected_destroy,
            repair_operators=selected_repair,
            improvement_operators=selected_improve,
        )

    def gym_space(self) -> spaces.Discrete:
        return self._inner.gym_space()

    def n(self) -> int:
        return int(self._inner.n())

    def id_to_action(self, action_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        d_local, r_local, i_local = self._inner.id_to_action(action_idx)
        return self._d_engine[int(d_local)], self._r_engine[int(r_local)], self._i_engine[int(i_local)]

    def action_to_id(
        self,
        destroy_idx: Optional[int],
        repair_idx: Optional[int],
        improvement_idx: Optional[int] = None,
    ) -> int:
        if destroy_idx is None or repair_idx is None or improvement_idx is None:
            raise ValueError("op_pair_mapped_v1 requires all three engine-global indices")
        try:
            local_d = self._d_engine.index(int(destroy_idx))
            local_r = self._r_engine.index(int(repair_idx))
            local_i = self._i_engine.index(int(improvement_idx))
        except ValueError as exc:
            raise ValueError("Operator indices not part of the mapped selection") from exc
        return self._inner.action_to_id(local_d, local_r, local_i)

    def sample(self) -> int:
        return int(self._inner.sample())
