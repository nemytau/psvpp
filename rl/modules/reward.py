"""Reward function implementations for the ALNS RL environment."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from rl.registries import register_reward


class BaseRewardFunction:
    """Base reward-function interface."""

    version: str = "base"

    def compute(self, result: Dict[str, Any], env: Any) -> float:  # pragma: no cover - interface only
        raise NotImplementedError


@register_reward("delta_cost_v3")
class DeltaCostReward(BaseRewardFunction):
    """Reward shaped around relative cost improvements and acceptance."""

    version = "3.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial_cost = env.initial_cost
        if not initial_cost or initial_cost <= 0:
            return 0.0

        current_cost = float(result.get("current_cost", initial_cost))
        best_cost = float(result.get("best_cost", current_cost))
        accepted = bool(result.get("accepted", False))
        is_new_best = bool(result.get("is_new_best", False))

        cost_improvement = (initial_cost - current_cost) / initial_cost
        best_improvement = (initial_cost - best_cost) / initial_cost

        improvement_reward = cost_improvement * 10.0
        best_reward = best_improvement * 15.0
        acceptance_bonus = 1.0 if accepted else -0.5
        new_best_bonus = 5.0 if is_new_best else 0.0
        progress_bonus = 0.1 if env.iteration > 0 else 0.0

        total_reward = (
            improvement_reward
            + best_reward
            + acceptance_bonus
            + new_best_bonus
            + progress_bonus
        )

        return float(np.clip(total_reward, -20.0, 50.0))


@register_reward("delta_cost_v4")
class DeltaCostStepDeltaReward(BaseRewardFunction):
    """Reward based on step-wise cost deltas (current + best)."""

    version = "4.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0

        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))

        cur = float(result.get("current_cost", prev_cur))
        best = float(result.get("best_cost", cur))

        denom_cur = max(abs(prev_cur), 1e-9)
        denom_best = max(abs(prev_best), 1e-9)

        delta_cur = (prev_cur - cur) / denom_cur
        delta_best = (prev_best - best) / denom_best

        if delta_cur < 0.0:
            delta_cur *= 0.3

        reward = 10.0 * delta_cur + 30.0 * delta_best

        acceptance_type = result.get("acceptance_type") or getattr(env, "last_acceptance_type", "unknown")
        is_new_best = bool(result.get("is_new_best", False))

        no_progress = (
            abs(cur - prev_cur) <= 1e-9
            and abs(best - prev_best) <= 1e-9
        )
        stagnation_count = int(getattr(env, "stagnation_count", 0) or 0)
        stagn_penalty = 0.0025 * min(stagnation_count, 200)

        made_progress = not no_progress

        if acceptance_type == "improvement":
            if made_progress:
                reward += 1.0
        elif acceptance_type == "annealing":
            if made_progress:
                reward += 0.2
        else:
            reward -= 0.5
        if is_new_best:
            reward += 5.0

        if no_progress:
            reward = -0.2 - stagn_penalty
        else:
            reward -= stagn_penalty

        if result.get("improvement_only_action") or result.get("improvement_operator_idx") is not None:
            reward -= 0.05

        return float(np.clip(reward, -5.0, 20.0))


@register_reward("delta_cost_v5")
class DeltaCostCandidateReward(BaseRewardFunction):
    """Reward from the candidate solution's own cost, independent of SA's
    stochastic accept/reject decision.

    delta_cost_v3's acceptance_bonus and delta_cost_v4's no_progress
    flattening both make reward magnitude partly a function of SA's random
    accept/reject coin flip rather than purely the proposed move's quality --
    a rejected candidate that nearly matched the current solution gets the
    exact same flat penalty as one that made things drastically worse, so the
    operator gets no gradient telling those two cases apart.

    This variant reads `result["candidate_cost"]` -- the solution's cost
    *before* SA's accept/reject decision runs (engine.rs computed this all
    along but never exposed it; the same key used to be silently aliased to
    the post-decision current_cost, see python_interface.rs) -- and computes
    reward purely from how that candidate compares to the previous current
    and best costs, regardless of what SA actually did with it. No
    accepted/acceptance_type branching at all.
    """

    version = "5.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0

        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))

        candidate = float(result.get("candidate_cost", result.get("current_cost", prev_cur)))

        denom_cur = max(abs(prev_cur), 1e-9)
        denom_best = max(abs(prev_best), 1e-9)

        delta_vs_current = (prev_cur - candidate) / denom_cur
        delta_vs_best = (prev_best - candidate) / denom_best

        # Same asymmetric dampening as delta_cost_v4: soften the downside so a
        # single bad candidate doesn't swamp the signal, without hiding it
        # entirely (unlike v4's no_progress flattening).
        if delta_vs_current < 0.0:
            delta_vs_current *= 0.3
        if delta_vs_best < 0.0:
            delta_vs_best *= 0.3

        reward = 10.0 * delta_vs_current + 30.0 * delta_vs_best

        if result.get("improvement_only_action") or result.get("improvement_operator_idx") is not None:
            reward -= 0.05

        return float(np.clip(reward, -5.0, 20.0))


@register_reward("delta_cost_v6")
class DeltaCostLateWeightedReward(BaseRewardFunction):
    """delta_cost_v5 with a bounded late-restart weighting multiplier.

    Motivation: operator-usage analysis (Exp1, op_triple_v2/op_pair_v2,
    2026-08-20/21) found PPO's entropy collapsing to near-zero early in
    training regardless of which operators are available or how large the
    action space is -- the policy settles on one action within the first
    1-2 iterations of a restart and never reconsiders it for the remaining
    ~298, even when that choice is barely improving anything (~1% over 300
    iterations for the dominant "stuck" branch observed on op_triple_v2
    medium, vs ~22% for the branch that happened to switch away from it).
    The flat per-step reward gives the policy no signal that continuing to
    search has more value late in a restart than early -- a fixed-size
    improvement at iteration 5 and the same improvement at iteration 295
    score identically, even though intensification only pays off if the
    policy is still willing to search for it late.

    This multiplies delta_cost_v5's improvement-driven terms by a weight
    that ramps from 1x (restart start) to 2x (restart end), using the same
    progress_horizon-normalized `progress` value the state features already
    use (rl/modules/state.py), so training and eval agree on what "late in
    the restart" means. Deliberately a bounded multiplicative ramp, not a
    division by iterations-left -- dividing blows up to infinity as a
    restart ends, letting the last few steps dominate the loss with reward
    spikes; multiplying by (1 + progress) only ever doubles the incentive,
    never lets late steps swamp everything else.
    """

    version = "6.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0

        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))

        candidate = float(result.get("candidate_cost", result.get("current_cost", prev_cur)))

        denom_cur = max(abs(prev_cur), 1e-9)
        denom_best = max(abs(prev_best), 1e-9)

        delta_vs_current = (prev_cur - candidate) / denom_cur
        delta_vs_best = (prev_best - candidate) / denom_best

        if delta_vs_current < 0.0:
            delta_vs_current *= 0.3
        if delta_vs_best < 0.0:
            delta_vs_best *= 0.3

        horizon = getattr(env, "progress_horizon", None) or getattr(env, "max_iterations", 1)
        progress = min(max(float(getattr(env, "iteration", 0)) / max(float(horizon), 1.0), 0.0), 1.0)
        late_weight = 1.0 + progress

        reward = late_weight * (10.0 * delta_vs_current + 30.0 * delta_vs_best)

        if result.get("improvement_only_action") or result.get("improvement_operator_idx") is not None:
            reward -= 0.05

        return float(np.clip(reward, -5.0, 20.0))


@register_reward("delta_cost_v7")
class DeltaCostLateWeightedNoOpPenaltyReward(DeltaCostLateWeightedReward):
    """delta_cost_v6 plus a penalty for *repeating* an improvement operator that just
    found nothing -- not for improvement-only steps finding nothing in general.

    Motivation: `improvement_chain_v2` operator-usage analysis (2026-08-25) found the
    trained policy always chains exactly `chain_max_consecutive_improvements` calls to
    the same run-to-convergence operator (`deep_relocation_v10`) before being forced
    back to a backbone step -- and positions 2+ in every single chain (98,550/98,550)
    produced literally zero cost change, since a run-to-convergence operator has
    nothing left to find on an unchanged solution. The flat `-0.05` improvement-action
    tax `delta_cost_v6` already applies doesn't distinguish "found nothing" from "found
    something," so the policy had no signal telling it repeating a just-converged
    operator is pointless.

    Important distinction: this should only punish the *foreseeable* waste of calling
    the same operator again right after it already ran (whether or not that prior call
    itself succeeded -- a run-to-convergence operator exhausts its neighborhood in one
    call regardless), not every no-op improvement-only step. Trying VNR after a fresh
    backbone step and finding nothing is a legitimate, unpredictable outcome of
    exploration and should not be punished the same as mechanically repeating the
    operator that just ran. So the gate is "same operator index as the immediately
    preceding step AND this call found nothing" -- not "found nothing" alone.

    `result["improvement_operator_idx"]` is already present in the raw result at
    reward-computation time (confirmed directly: it comes from the Rust engine's own
    return value, not later Python enrichment). `env.last_improvement_idx` still holds
    the *previous* step's operator index when `compute()` runs -- the environment only
    updates it to the current step's value afterward -- and defaults to -1.0 whenever
    the previous step wasn't a pure improvement-only step at all (e.g. a backbone step
    just ran), so a fresh attempt right after backbone is never treated as a repeat.

    `result["action_type"]` (set by the environment before reward computation runs) is
    `"chain_improvement_only"` or `"improvement_only"` only when a step ran an
    improvement operator with *no* destroy/repair at all -- i.e. only the
    improvement_chain_v1/v2-style pure-improvement steps this diagnosis is about.
    (`result["improvement_only_action"]` looks like the right field for this but is a
    trap: the environment sets it *after* calling the reward function, so it is never
    actually populated yet when `compute()` runs -- confirmed by the first version of
    this reward silently never firing at all.) `op_triple_v2`-style steps always carry
    an explicit destroy/repair pair, so `action_type` is never one of these two values
    there and this penalty never fires -- a D×R×I step where the improvement component
    happens to find nothing (destroy/repair already did the real work) is not the same
    failure mode and shouldn't be punished the same way.

    Penalty is a flat -1.0, matching the rough magnitude of a genuine success's reward
    in this reward's own scale (~+1.0 for a real improving candidate) -- "repeat the
    same operator and get nothing" should read as clearly bad as "call an operator and
    get a real gain" reads as good, so switching away is a clear improvement rather
    than a marginal one.
    """

    version = "7.0"

    NO_OP_IMPROVEMENT_PENALTY = 1.0

    def _no_progress(self, result: Dict[str, Any], env: Any) -> bool:
        """True if this improvement-only step made no real progress.

        v7's definition: the proposed candidate is numerically identical to the
        previous current/best cost. This catches a true run-to-convergence no-op
        (e.g. deep_relocation_v10 finding nothing at all) but misses an operator
        that proposes a *different*, cost-increasing candidate that then gets
        rejected -- see v8, which fixes this using `accepted` directly.
        """
        initial = env.initial_cost or result.get("initial_cost", 0.0) or 1.0
        prev_cur = float(getattr(env, "last_current_cost", initial))
        prev_best = float(getattr(env, "last_best_cost", initial))
        candidate = float(result.get("candidate_cost", result.get("current_cost", prev_cur)))
        return abs(prev_cur - candidate) < 1e-9 and abs(prev_best - candidate) < 1e-9

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        reward = super().compute(result, env)

        if result.get("action_type") in ("chain_improvement_only", "improvement_only"):
            current_op_idx = result.get("improvement_operator_idx")
            prev_op_idx = getattr(env, "last_improvement_idx", -1.0)
            same_operator_as_last_step = (
                current_op_idx is not None
                and prev_op_idx is not None
                and prev_op_idx >= 0.0
                and int(current_op_idx) == int(prev_op_idx)
            )

            if same_operator_as_last_step and self._no_progress(result, env):
                reward -= self.NO_OP_IMPROVEMENT_PENALTY

        return float(np.clip(reward, -5.0, 20.0))


@register_reward("delta_cost_v8")
class DeltaCostAcceptedNoOpPenaltyReward(DeltaCostLateWeightedNoOpPenaltyReward):
    """delta_cost_v7 with a corrected "no progress" definition.

    Analysis of v7's own results (2026-08-25) found `voyage_number_reduction_v1`
    (VNR) repeats slipping past the penalty even though they were just as wasteful
    as `deep_relocation_v10`'s repeats: VNR's Rust implementation
    (voyage_number_reduction.rs) applies its "best" relocation as soon as it's
    *feasible*, without checking that the resulting cost_increase is actually
    negative -- so VNR can propose eliminating a voyage at a net cost increase.
    The environment correctly rejects that candidate (`accepted=False`), so the
    real solution never changes -- but v7's no-progress check compared the
    *candidate's* cost to the previous cost, and a rejected-but-different
    candidate reads as "progress" under that check even though it changed
    nothing real. Traced one concrete case: the identical cost_before=334066.05
    -> cost_after=336291.65 pair (a cost *increase*) recurred at iterations 6,
    7, 10, 11 of one restart, rejected every single time -- the same losing
    trade proposed over and over, exactly the pattern this reward exists to
    penalize, but v7 missed it.

    Fix: use `result["accepted"]` directly instead of comparing candidate cost
    to previous cost. The environment already computes this correctly for
    exactly this purpose -- a literally-unchanged candidate is explicitly
    downgraded to `accepted=False` too (see rl_alns_environment.py's
    `rejected_no_change` handling), so `not accepted` covers both a true no-op
    and a rejected-worse-candidate symmetrically, for any operator.
    """

    version = "8.0"

    def _no_progress(self, result: Dict[str, Any], env: Any) -> bool:
        return not bool(result.get("accepted", False))


@register_reward("delta_cost_v9")
class DeltaCostTimePenaltyReward(DeltaCostAcceptedNoOpPenaltyReward):
    """delta_cost_v8 plus a flat time-cost tax, independent of the step's outcome.

    Motivation: operator-timing samples (2026-08-25, both sizes) found an
    enormous spread in wall-clock cost per operator call -- e.g.
    `fleet_and_cost_reduction_kisialiou_v1` averages ~0.31ms on medium while
    `deep_swap_v10` averages ~159ms, roughly 500x slower, for whatever quality
    gain it happens to produce. Nothing in the reward so far accounts for
    this at all -- delta_cost_v3 through v8 are purely a function of solution
    quality, so a policy has no reason to prefer a fast operator over a slow
    one that delivers the same cost improvement.

    This is deliberately additive (`reward - k * elapsed_ms`), not
    multiplicative (`reward * reference_ms / elapsed_ms`). A multiplicative
    scaling has a sign problem: for a *negative* reward (a failure, or the
    no-op penalty from v7/v8), multiplying by a small time-weight for a slow
    operator would *shrink* the penalty -- a slow failure would read as less
    bad than a fast one, which is backwards. Subtracting a flat time cost
    keeps speed as an independent axis: a fast success keeps almost all of
    its positive reward; a slow failure is punished by both the existing
    failure penalty *and* the added time tax, and a slow success has its
    reward reduced but not inverted.

    `k = 0.005` (per millisecond), with the time penalty itself capped at 3.0,
    calibrated against real timing samples -- and one live check that caught
    the first calibration attempt (k=0.01, uncapped) being badly miscalibrated:
    `deep_swap_v10`'s runtime is highly variable (2+ orders of magnitude across
    calls, not just across operators), and one genuinely successful call
    (accepted=True, cost improved, v8's own reward = +0.31) took 644ms --
    4x the ~159ms average sampled on medium. At k=0.01 uncapped that alone
    produced a penalty of ~6.4, driving the reward to the -5.0 clip floor and
    completely erasing the fact that the move succeeded -- indistinguishable
    from a slow *failure*. That teaches "never use this operator," not "use it
    only when the gain justifies the cost."

    With k=0.005 and a 3.0 cap: negligible (<=0.005) for anything under ~1ms
    (first_fit_insertion, fleet_and_cost_reduction, most destroy/repair
    pairs), modest (~0.05-0.13) for deep_relocation_v10, meaningful but not
    overwhelming (~0.8, comparable to a typical success reward rather than
    dwarfing it) for deep_swap_v10's ~159ms average on medium, and the same
    644ms outlier case now lands around 0.31 - 3.0 = -2.69 -- clearly bad, but
    distinguishable from a failed-and-slow case rather than identical to it.

    Uses `result["elapsed_ns"]`, confirmed present at reward-compute time and
    equal to the operator's own execution time for both backbone steps
    (destroy+repair combined) and pure improvement-only steps, so no separate
    handling is needed per action type.
    """

    version = "9.0"

    TIME_PENALTY_PER_MS = 0.005
    MAX_TIME_PENALTY = 3.0

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        reward = super().compute(result, env)

        elapsed_ns = result.get("elapsed_ns")
        if elapsed_ns is not None and elapsed_ns > 0:
            elapsed_ms = float(elapsed_ns) / 1e6
            time_penalty = min(self.TIME_PENALTY_PER_MS * elapsed_ms, self.MAX_TIME_PENALTY)
            reward -= time_penalty

        return float(np.clip(reward, -5.0, 20.0))


# ---------------------------------------------------------------------------
# NORM 2026 reward-design ablation: simple, ALNS-style rewards.
# ---------------------------------------------------------------------------


def _is_new_best(result: Dict[str, Any], env: Any) -> bool:
    """True if this step improved the restart's best cost.

    Compared against env.last_best_cost, which the environment still holds at the
    *previous* best when compute() runs (it is updated right after the reward call).
    """
    prev_best = getattr(env, "last_best_cost", None)
    if prev_best is None:
        return bool(result.get("is_new_best", False))
    return float(result.get("best_cost", prev_best)) < float(prev_best) - 1e-9


@register_reward("alns_score_v1")
class AlnsScoreReward(BaseRewardFunction):
    """Classic ALNS operator score used as the RL reward (Ropke & Pisinger, 2006).

    sigma1 = 33 for a new global best, sigma2 = 9 for a solution better than the current
    one, sigma3 = 3 for an accepted worse solution (simulated-annealing acceptance), 0 for a
    rejected one -- rescaled so a new best is worth 1.0. Uses the environment's
    acceptance_type ("improvement" = accepted and better than current), which is set before
    the reward is computed.
    """

    version = "alns-score-1.0"

    NEW_BEST = 1.0
    BETTER = 9.0 / 33.0
    ACCEPTED_WORSE = 3.0 / 33.0

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        if _is_new_best(result, env):
            return self.NEW_BEST
        acceptance_type = result.get("acceptance_type")
        if acceptance_type == "improvement":
            return self.BETTER
        if acceptance_type == "annealing":
            return self.ACCEPTED_WORSE
        return 0.0


@register_reward("alns_score_time_v1")
class AlnsScoreTimeReward(AlnsScoreReward):
    """alns_score_v1 minus a capped, operator-time tax (same additive design as
    delta_cost_v9, rescaled to the 0..1 score range).

    0.002 per ms, capped at 0.5: negligible for cheap steps (<1 ms), ~0.3 for an unlimited
    deep_swap call on medium (~160 ms), so a slow step that finds a new best still nets
    +0.5 or more, while a slow step that finds nothing is clearly negative.
    """

    version = "alns-score-time-1.0"

    TIME_PENALTY_PER_MS = 0.002
    MAX_TIME_PENALTY = 0.5

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        reward = super().compute(result, env)
        elapsed_ns = result.get("elapsed_ns")
        if elapsed_ns is not None and elapsed_ns > 0:
            reward -= min(self.TIME_PENALTY_PER_MS * float(elapsed_ns) / 1e6, self.MAX_TIME_PENALTY)
        return float(reward)


@register_reward("new_best_v1")
class NewBestReward(BaseRewardFunction):
    """Sparse: +1 only when a step finds a new best solution (what evaluation measures)."""

    version = "new-best-1.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        return 1.0 if _is_new_best(result, env) else 0.0


@register_reward("best_gain_v1")
class BestGainReward(BaseRewardFunction):
    """Relative improvement of the best cost, in percent (0 if the best did not improve).

    Magnitude-aware version of new_best_v1; clipped at 20 so the very first steps of a
    restart (large gains from the constructive solution) cannot dominate.
    """

    version = "best-gain-1.0"

    def compute(self, result: Dict[str, Any], env: Any) -> float:
        prev_best = getattr(env, "last_best_cost", None)
        if prev_best is None or float(prev_best) <= 0.0:
            return 0.0
        best = float(result.get("best_cost", prev_best))
        gain = (float(prev_best) - best) / float(prev_best) * 100.0
        return float(np.clip(gain, 0.0, 20.0))
