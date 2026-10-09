"""Where one ALNS step's time goes, per method: engine search vs. Python overhead.

Runs RL (a trained PPO model), the adaptive baseline and Kisialiou on one test instance, one
job alone on the machine, and splits each step's time into
  - search: CPU time inside the Rust engine (env.search_cpu_seconds),
  - Python inside env.step (state encoding, reward, operator-log row, result conversion),
  - policy inference (SB3 predict, RL only),
  - environment creation / reset / close, per step.
It also times the policy decision on its own (SB3 predict, lean PyTorch, NumPy) and the
engine's solution-metrics routine (a conservative upper bound for native state construction).

Produced the per-step table and the inference / state-cost numbers in docs/EXPERIMENTS.md,
"Timing protocol and RL decision overhead" (2026-10-09, medium, Exp 8 medium model).

    PYTHONPATH=. python scripts/timing/rl_step_overhead.py --size medium \\
        --model output/<run>/model/ppo_rl_improvement_mvp.zip --log-dir /tmp/rl_step_overhead_logs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO  # noqa: E402

from rl.dataset_manager import GeneratedDatasetManager  # noqa: E402
from rl.rl_alns_environment import ALNSEnvironment  # noqa: E402


def make_env(path: str, size: str, mode: str, restart: int, log_dir: str) -> ALNSEnvironment:
    return ALNSEnvironment(
        problem_instance=path, max_iterations=300, progress_horizon=300, seed=100 + restart,
        problem_instance_paths=[path], problem_sampling_strategy="round_robin",
        enable_operator_logging=True, operator_logging_mode="reinforcement_learning",
        operator_logging_mode_label="timing", operator_logging_size=size,
        operator_logging_dataset_name=f"timing_{mode}", operator_logging_restart_id=restart,
        operator_logging_stable_filename=True, operator_logging_dir=log_dir,
        action_module="op_triple_v2" if mode == "rl" else "adaptive_weighted_v1",
        state_module="features_improvement_v3" if mode == "rl" else "features_v3",
        reward_module="delta_cost_v9",
        algorithm_mode="kisialiou" if mode == "kisialiou" else "baseline",
        size_bucket="small" if size == "small" else "large",
    )


def step_breakdown(model: PPO, path: str, size: str, log_dir: str) -> None:
    for mode, restarts, steps in (("rl", 4, 300), ("baseline", 2, 300), ("kisialiou", 1, 30)):
        t = dict(create=0.0, predict=0.0, step=0.0, search=0.0, n=0)
        for k in range(restarts):
            a = time.perf_counter()
            env = make_env(path, size, mode, k, log_dir)
            obs, _ = env.reset(seed=100 + k, options={"problem_path": path})
            t["create"] += time.perf_counter() - a
            for _ in range(steps):
                a = time.perf_counter()
                action = int(model.predict(obs, deterministic=True)[0]) if mode == "rl" else 0
                b = time.perf_counter()
                obs, *_ = env.step(action)
                t["predict"] += b - a
                t["step"] += time.perf_counter() - b
                t["n"] += 1
            t["search"] += env.search_cpu_seconds
            a = time.perf_counter()
            env.close()
            t["create"] += time.perf_counter() - a
        n = t["n"]
        total = t["create"] + t["predict"] + t["step"]
        print(f"{mode:9s} per step: search {t['search'] / n * 1e3:7.2f} ms | Python in env.step "
              f"{(t['step'] - t['search']) / n * 1e3:5.2f} ms | policy inference {t['predict'] / n * 1e3:5.2f} ms | "
              f"env create/reset/close {t['create'] / n * 1e3:5.2f} ms | search share {t['search'] / total:4.0%}")


def decision_costs(model: PPO, path: str, size: str, log_dir: str) -> None:
    import torch

    pol = model.policy
    obs = np.random.rand(pol.observation_space.shape[0]).astype(np.float32)
    n = 2000
    t = time.perf_counter()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    sb3 = (time.perf_counter() - t) / n
    torch.set_num_threads(1)
    with torch.no_grad():
        x = torch.as_tensor(obs).unsqueeze(0)
        t = time.perf_counter()
        for _ in range(n):
            int(pol.action_net(pol.mlp_extractor.forward_actor(x)).argmax())
        lean_torch = (time.perf_counter() - t) / n
    weights = [(m.weight.detach().numpy(), m.bias.detach().numpy())
               for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    w_out, b_out = pol.action_net.weight.detach().numpy(), pol.action_net.bias.detach().numpy()
    t = time.perf_counter()
    for _ in range(n):
        h = obs
        for w, b in weights:
            h = np.tanh(w @ h + b)
        int(np.argmax(w_out @ h + b_out))
    lean_np = (time.perf_counter() - t) / n
    print(f"policy decision: SB3 predict {sb3 * 1e6:.0f} us | lean PyTorch {lean_torch * 1e6:.0f} us | "
          f"NumPy {lean_np * 1e6:.0f} us  (compiled: scripts/timing/mlp_decision_cost.rs)")

    env = make_env(path, size, "rl", 0, log_dir)
    env.reset(seed=1, options={"problem_path": path})
    for _ in range(20):
        env.step(29)
    t = time.perf_counter()
    for _ in range(n):
        env.alns.extract_solution_metrics()
    print(f"engine solution-metrics call (upper bound for native state construction): "
          f"{(time.perf_counter() - t) / n * 1e6:.1f} us")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", default="medium", choices=("small", "medium", "large"))
    ap.add_argument("--model", required=True, help="Trained op_triple_v2 / features_improvement_v3 PPO model")
    ap.add_argument("--log-dir", default="/tmp/rl_step_overhead_logs", help="Where the operator logs go")
    args = ap.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    path = GeneratedDatasetManager().prepare_size(args.size)["test"][0]
    model = PPO.load(args.model)
    step_breakdown(model, path, args.size, args.log_dir)
    decision_costs(model, path, args.size, args.log_dir)


if __name__ == "__main__":
    main()
