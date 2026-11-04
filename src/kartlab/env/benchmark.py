"""Environment throughput: one env (random or autopilot policy) and vectorised envs
(DummyVecEnv vs SubprocVecEnv).

Run with ``uv run python scripts/benchmark_env.py`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.sim.autopilot import Autopilot


def make_env(config: EnvConfig, seed: int) -> Callable[[], KartEnv]:
    def _init() -> KartEnv:
        env = KartEnv(config)
        env.reset(seed=seed)
        return env

    return _init


def bench_single(config: EnvConfig, policy: str, steps: int, seed: int = 0) -> dict:
    """Steps per second of one env, including resets, with a random or autopilot policy."""
    env = KartEnv(config)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    pilot: Autopilot | None = None
    episodes = 0
    start = time.perf_counter()
    for _ in range(steps):
        if policy == "autopilot":
            if pilot is None or pilot.track is not env.track:
                pilot = Autopilot(env.track)
            action = np.array(pilot.act(env.racer), dtype=np.float32)
        else:
            action = rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes += 1
            env.reset()
    elapsed = time.perf_counter() - start
    return {
        "policy": policy,
        "steps": steps,
        "steps_per_s": steps / elapsed,
        "episodes": episodes,
        "mean_episode_steps": steps / max(episodes, 1),
    }


def bench_reset(config: EnvConfig, count: int, seed: int = 0) -> float:
    """Mean seconds per reset (a new random track is generated when ``track`` is None)."""
    env = KartEnv(config)
    env.reset(seed=seed)
    start = time.perf_counter()
    for _ in range(count):
        env.reset()
    return (time.perf_counter() - start) / count


def bench_vec(config: EnvConfig, kind: str, n_envs: int, steps: int, seed: int = 0) -> dict:
    """Total env steps per second of a vectorised env with random actions (startup excluded)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env(config, seed + i) for i in range(n_envs)]
    vec = DummyVecEnv(fns) if kind == "dummy" else SubprocVecEnv(fns)
    try:
        rng = np.random.default_rng(seed)
        vec.reset()
        for _ in range(50):  # warm-up
            vec.step(rng.uniform(-1.0, 1.0, (n_envs, 2)).astype(np.float32))
        start = time.perf_counter()
        for _ in range(steps):
            vec.step(rng.uniform(-1.0, 1.0, (n_envs, 2)).astype(np.float32))
        elapsed = time.perf_counter() - start
    finally:
        vec.close()
    return {"kind": kind, "n_envs": n_envs, "steps_per_s": steps * n_envs / elapsed}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mide los pasos por segundo del entorno.")
    parser.add_argument("--steps", type=int, default=6000, help="pasos con un solo entorno")
    parser.add_argument("--resets", type=int, default=40, help="reinicios para medir su coste")
    parser.add_argument(
        "--vec-steps", type=int, default=1500, help="pasos de cada VecEnv (por entorno)"
    )
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4, 6, 8])
    parser.add_argument("--no-vec", action="store_true", help="solo el entorno individual")
    parser.add_argument("--json", type=Path, help="guarda los resultados en este JSON")
    args = parser.parse_args(argv)

    config = EnvConfig()
    results: dict = {"machine": platform.processor() or platform.machine(), "single": []}
    print(f"Equipo: {results['machine']}")
    for policy in ("random", "autopilot"):
        row = bench_single(config, policy, args.steps)
        results["single"].append(row)
        print(
            f"1 entorno, política {policy:9s}: {row['steps_per_s']:7.0f} pasos/s "
            f"({row['episodes']} episodios, {row['mean_episode_steps']:.0f} pasos de media)"
        )
    results["reset_ms"] = bench_reset(config, args.resets) * 1000
    print(f"Reinicio con circuito nuevo: {results['reset_ms']:.1f} ms")

    if not args.no_vec:
        results["vec"] = []
        print(f"\n{'VecEnv':>14} {'entornos':>9} {'pasos/s':>9}")
        for kind in ("dummy", "subproc"):
            for n in args.envs:
                row = bench_vec(config, kind, n, args.vec_steps)
                results["vec"].append(row)
                print(f"{kind:>14} {n:>9} {row['steps_per_s']:>9.0f}")
        best = max(results["vec"], key=lambda r: r["steps_per_s"])
        print(f"\nMás rápido: {best['kind']} con {best['n_envs']} entornos")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Resultados en {args.json}")
