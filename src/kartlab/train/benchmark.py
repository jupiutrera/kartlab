"""Training throughput with PPO in the loop (collection plus updates).

Compares DummyVecEnv and SubprocVecEnv with 4, 6 and 8 envs, then tries more PyTorch
threads on the fastest combination. Run with ``uv run python scripts/benchmark_train.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from kartlab.common.system import limit_child_threads
from kartlab.train.config import PPOParams, TrainConfig
from kartlab.train.train import build_model, build_vec_env


def bench_ppo(kind: str, n_envs: int, torch_threads: int, rollouts: int, n_steps: int) -> dict:
    import torch

    config = TrainConfig(
        name="benchmark",
        n_envs=n_envs,
        vec_env=kind,
        torch_threads=torch_threads,
        low_priority=False,
        ppo=dataclasses.replace(PPOParams(), n_steps=n_steps),
    )
    torch.set_num_threads(torch_threads)
    limit_child_threads(1)
    vec_env = build_vec_env(config)
    try:
        model = build_model(config, vec_env, None)
        per_rollout = n_steps * n_envs
        model.learn(per_rollout)  # warm-up: first rollout and update
        start = time.perf_counter()
        model.learn(rollouts * per_rollout, reset_num_timesteps=False)
        elapsed = time.perf_counter() - start
    finally:
        vec_env.close()
    return {
        "kind": kind,
        "n_envs": n_envs,
        "torch_threads": torch_threads,
        "steps_per_s": rollouts * per_rollout / elapsed,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mide los pasos por segundo entrenando con PPO.")
    parser.add_argument("--envs", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=PPOParams.n_steps)
    parser.add_argument("--json", type=Path, help="guarda los resultados en este JSON")
    args = parser.parse_args(argv)

    rows = []
    print(f"{'VecEnv':>8} {'entornos':>9} {'hilos':>6} {'pasos/s':>9}", flush=True)

    def run(kind, n, threads):
        row = bench_ppo(kind, n, threads, args.rollouts, args.n_steps)
        rows.append(row)
        print(f"{kind:>8} {n:>9} {threads:>6} {row['steps_per_s']:>9.0f}", flush=True)

    for kind in ("dummy", "subproc"):
        for n in args.envs:
            run(kind, n, 1)
    best = max(rows, key=lambda r: r["steps_per_s"])
    for threads in args.threads:
        run(best["kind"], best["n_envs"], threads)
    best = max(rows, key=lambda r: r["steps_per_s"])
    print(
        f"\nMás rápido: {best['kind']} con {best['n_envs']} entornos y "
        f"{best['torch_threads']} hilos de PyTorch ({best['steps_per_s']:.0f} pasos/s)"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "best": best}, indent=2), encoding="utf-8")
