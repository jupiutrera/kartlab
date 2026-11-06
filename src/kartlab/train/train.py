"""PPO training on CPU with vectorised envs.

Run with ``uv run python scripts/train.py configs/<config>.yaml`` (``--help`` for options).
Environment workers start with spawn on Windows, so nothing heavy happens at import time.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from kartlab.common.io import write_yaml_atomic
from kartlab.common.system import limit_child_threads, lower_priority
from kartlab.env.kart_env import EnvConfig, KartEnv
from kartlab.train.config import TrainConfig
from kartlab.train.run import RUNS_DIR, RunPaths, create_run


def make_env(config: EnvConfig):
    def _init() -> KartEnv:
        return KartEnv(config)

    return _init


def build_vec_env(config: TrainConfig):
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
        VecNormalize,
    )

    env_config = config.env_config()
    fns = [make_env(env_config) for _ in range(config.n_envs)]
    vec = SubprocVecEnv(fns) if config.vec_env == "subproc" else DummyVecEnv(fns)
    vec = VecMonitor(vec, info_keywords=("laps",))
    return VecNormalize(
        vec,
        norm_obs=config.normalize_obs,
        norm_reward=config.normalize_reward,
        gamma=config.ppo.gamma,
    )


def build_model(config: TrainConfig, vec_env, tensorboard: Path | None):
    from stable_baselines3 import PPO

    ppo = dataclasses.asdict(config.ppo)
    arch = list(ppo.pop("net_arch"))
    log_std_init = ppo.pop("log_std_init")
    if ppo.pop("lr_schedule") == "linear":
        initial = ppo["learning_rate"]
        ppo["learning_rate"] = lambda remaining: initial * remaining  # 1 -> 0 over the run
    return PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}, "log_std_init": log_std_init},
        tensorboard_log=str(tensorboard) if tensorboard else None,
        seed=config.seed,
        device="cpu",
        verbose=0,
        **ppo,
    )


def train(
    config: TrainConfig,
    runs_dir: Path = RUNS_DIR,
    verbose: int = 1,
    stop_file: Path | None = None,
) -> RunPaths:
    import torch
    from stable_baselines3.common.callbacks import CallbackList

    from kartlab.train.callbacks import (
        CheckpointCallback,
        CurriculumCallback,
        EvalCallback,
        MetricsCallback,
        StopFileCallback,
    )

    if config.low_priority:
        lower_priority()
    limit_child_threads(1)
    torch.set_num_threads(config.torch_threads)

    run = create_run(config.to_dict(), config.name, runs_dir)
    if verbose:
        print(f"Run: {run.root}", flush=True)
    vec_env = build_vec_env(config)
    model = build_model(config, vec_env, run.tensorboard)
    checkpoints = CheckpointCallback(run, config.checkpoint_every, config.checkpoint_at)
    metrics = MetricsCallback(run, verbose)
    evaluation = EvalCallback(
        run, config.eval_every, config.eval_tracks, config.env_config(), config.eval_laps, verbose
    )
    callbacks = [metrics, checkpoints, evaluation]
    stopper = StopFileCallback(stop_file) if stop_file is not None else None
    if stopper is not None:
        callbacks.append(stopper)
    if config.curriculum:
        callbacks.append(CurriculumCallback(config.curriculum, verbose))

    start = time.perf_counter()
    interrupted = False
    try:
        model.learn(config.total_steps, callback=CallbackList(callbacks), tb_log_name="ppo")
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("Interrumpido: guardando el último checkpoint", flush=True)
    finally:
        stopped = stopper is not None and stopper.stopped
        if stopped and verbose:
            print("Parado desde el menú: guardando el último checkpoint", flush=True)
        checkpoints.save()
        if not (interrupted or stopped):  # a stop is a deliberate cut: no final evaluation
            evaluation.run_eval()
        vec_env.close()

    best = evaluation.best_result
    write_yaml_atomic(
        run.summary,
        {
            "steps": int(model.num_timesteps),
            "seconds": round(time.perf_counter() - start, 1),
            "interrupted": interrupted or stopped,
            "first_lap": metrics.first_lap,
            "best_eval": None
            if best is None
            else {"success_rate": best.success_rate, "progress_mean": best.progress_mean},
        },
    )
    return run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena un agente PPO.")
    parser.add_argument("config", type=Path, help="fichero YAML de configs/")
    parser.add_argument("--name", help="nombre del run (por defecto, el de la configuración)")
    parser.add_argument("--steps", type=int, help="pasos totales (sustituye a total_steps)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="si este fichero existe, se para guardando un checkpoint (lo usa el menú)",
    )
    args = parser.parse_args(argv)

    config = TrainConfig.from_yaml(args.config)
    overrides = {}
    if args.name:
        overrides["name"] = args.name
    if args.steps:
        overrides["total_steps"] = args.steps
    config = dataclasses.replace(config, **overrides)
    run = train(config, args.runs_dir, stop_file=args.stop_file)
    print(f"Terminado: {run.root}")


if __name__ == "__main__":  # python -m kartlab.train.train (the main menu launches it so)
    main()
