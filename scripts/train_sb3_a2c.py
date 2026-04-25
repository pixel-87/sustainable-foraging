#!/usr/bin/env python3
"""Train an A2C agent on the Sustainable Foraging AEC environment via SB3.

Uses the same environment wrapping and metrics collection as ``train_sb3.py``
(PPO) so that the results are directly comparable on the dashboard.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import supersuit as ss
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import get_preset, get_training_defaults
from pettingzoo.utils import aec_to_parallel
from scripts._bench_utils import MetricsTracker, get_standard_parser, save_experiment_config


# ---------------------------------------------------------------------------
# SB3 callback that delegates to the shared MetricsTracker
# ---------------------------------------------------------------------------


class _SB3MetricsCallback(BaseCallback):
    """Collect per-episode metrics and write to :class:`MetricsTracker` CSV."""

    def __init__(self, tracker: MetricsTracker, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._tracker: MetricsTracker = tracker

        # Per vec-env slot accumulators (initialised in _on_training_start)
        self._slots: list[dict[str, Any]] = []

    # -- lifecycle ----------------------------------------------------------

    def _on_training_start(self) -> None:
        n_envs: int = self.training_env.num_envs
        self._slots = [self._empty_slot() for _ in range(n_envs)]

    def _on_training_end(self) -> None:
        self._tracker.close()

    # -- per-step -----------------------------------------------------------

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [{}] * len(rewards))

        for i in range(len(rewards)):
            s = self._slots[i]
            s["reward"] += float(rewards[i])
            s["length"] += 1

            info: dict[str, Any] = infos[i] if i < len(infos) else {}
            s["foods_collected"] += int(info.get("foods_collected", 0))
            s["cooperative_collections"] += int(info.get("cooperative_collections", 0))
            s["solo_collections"] += int(info.get("solo_collections", 0))
            s["failed_loads"] += int(info.get("failed_loads", 0))
            s["collisions"] += int(info.get("collisions", 0))
            s["food_remaining"] = int(info.get("food_remaining", 0))

            for act_name, count in info.get("action_counts", {}).items():
                s["action_counts"][str(act_name)] += int(count)

            for j, r in enumerate(info.get("per_agent_rewards", [])):
                s["agent_rewards"][f"agent_{j}"] += float(r)

            if dones[i]:
                self._flush(i)

        return True

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _empty_slot() -> dict[str, Any]:
        return {
            "reward": 0.0,
            "length": 0,
            "foods_collected": 0,
            "cooperative_collections": 0,
            "solo_collections": 0,
            "failed_loads": 0,
            "collisions": 0,
            "food_remaining": 0,
            "action_counts": defaultdict(int),
            "agent_rewards": defaultdict(float),
        }

    def _flush(self, slot: int) -> None:
        s = self._slots[slot]
        self._tracker.on_episode_end(
            self.num_timesteps,
            {
                "reward_total": s["reward"],
                "length": s["length"],
                "foods_collected": s["foods_collected"],
                "cooperative_collections": s["cooperative_collections"],
                "solo_collections": s["solo_collections"],
                "failed_loads": s["failed_loads"],
                "food_remaining_end": s["food_remaining"],
                "collisions": s["collisions"],
                "action_counts": dict(s["action_counts"]),
                "agent_rewards": dict(s["agent_rewards"]),
            },
        )

        # Log to SB3's TensorBoard logger as well
        self.logger.record("episode/reward", s["reward"])
        self.logger.record("episode/length", s["length"])
        self.logger.record("episode/foods_collected", s["foods_collected"])
        self.logger.record("episode/count", self._tracker.episode_count)

        self._slots[slot] = self._empty_slot()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(config: dict[str, Any] | None = None, num_envs: int = 1) -> Any:
    """Create and wrap the foraging environment for SB3 training."""
    cfg: dict[str, Any] = config or get_preset("fair")
    env = AECForagingEnv(**cfg)
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=0, base_class="stable_baselines3"
    )
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    run_name: str = args.name or f"sb3_a2c_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir: Path = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir: Path = log_dir / "tb"
    model_path: Path = log_dir / "model"

    env_config: dict[str, Any] = get_preset(args.preset)

    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="A2C",
        library="SB3",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        policy="MlpPolicy",
    )

    print(f"Run name   : {run_name}")
    print(f"Preset     : {args.preset}")
    print(f"Log dir    : {log_dir}")
    print(f"Timesteps  : {args.timesteps:,}")
    print(f"LR         : {args.lr}")
    print(f"Num Envs   : {args.num_envs}")
    print()

    env = make_env(env_config, num_envs=args.num_envs)

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=args.gamma,
        learning_rate=args.lr,
        tensorboard_log=str(tb_dir),
    )

    tracker = MetricsTracker(log_dir / "metrics.csv")
    callback = _SB3MetricsCallback(tracker)

    print("Starting A2C training...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    model.save(str(model_path))
    print(f"  Model saved to: {model_path}.zip")

    # Quick evaluation
    print("\nEvaluating trained model (10 episodes)...")
    env.close()
    env = make_env(env_config)

    obs = env.reset()
    eval_rewards: list[float] = []
    ep_reward: float = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)  # type: ignore[arg-type]
        obs, reward, done, info = env.step(action)  # type: ignore[arg-type]
        ep_reward += float(sum(reward))
        if any(done):
            eval_rewards.append(ep_reward)
            ep_reward = 0.0
            obs = env.reset()
            if len(eval_rewards) >= 10:
                break

    if eval_rewards:
        print(f"  Eval episodes  : {len(eval_rewards)}")
        print(f"  Mean reward    : {np.mean(eval_rewards):.4f}")
        print(f"  Std reward     : {np.std(eval_rewards):.4f}")

    env.close()
    print(f"\nDone! Visualize with:  uv run python -m scripts.compare_algorithms logs/{run_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_standard_parser(
        description="Train A2C (SB3) on Sustainable Foraging"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    train(parser.parse_args())
