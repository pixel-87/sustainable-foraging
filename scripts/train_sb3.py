#!/usr/bin/env python3
"""Train a PPO agent on the Level-Based Foraging AEC environment.

Logs training metrics to:
  - CSV file                 (./logs/<run_name>/metrics.csv)
  - Experiment config        (./logs/<run_name>/config.json)
  - TensorBoard events       (./logs/<run_name>/tb/)
  - Saved model              (./logs/<run_name>/model.zip)
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from lbforaging.foraging import AECForagingEnv
from lbforaging.foraging.sustainable_benchmark import (
    BENCHMARK_NAME,
    get_preset,
    list_presets,
)
from pettingzoo.utils import aec_to_parallel


# ---------------------------------------------------------------------------
# Custom callback for comprehensive data collection
# ---------------------------------------------------------------------------
class MetricsCallback(BaseCallback):
    """Collects per-episode multi-agent metrics and writes to CSV + TB."""

    CSV_COLUMNS = [
        "episode",
        "timestep",
        "wall_time",
        # Episode-level
        "reward_total",
        "length",
        # Food
        "foods_collected",
        "cooperative_collections",
        "solo_collections",
        "failed_loads",
        "food_remaining_end",
        # Collisions
        "collisions",
        # Action distribution
        "actions_north",
        "actions_south",
        "actions_east",
        "actions_west",
        "actions_load",
        "actions_none",
        # Per-agent rewards (agent_0, agent_1, ...)
        "agent_rewards",
    ]

    def __init__(self, csv_path: str, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path

        # Per vec-env slot accumulators
        self._ep_rewards = []
        self._ep_lengths = []
        self._ep_foods_collected = []
        self._ep_coop_collections = []
        self._ep_solo_collections = []
        self._ep_failed_loads = []
        self._ep_collisions = []
        self._ep_action_counts = []
        self._ep_agent_rewards = []
        self._ep_food_remaining = []

        self._ep_count = 0
        self._csv_writer = None
        self._csv_file = None

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self._ep_rewards = [0.0] * n_envs
        self._ep_lengths = [0] * n_envs
        self._ep_foods_collected = [0] * n_envs
        self._ep_coop_collections = [0] * n_envs
        self._ep_solo_collections = [0] * n_envs
        self._ep_failed_loads = [0] * n_envs
        self._ep_collisions = [0] * n_envs
        self._ep_action_counts = [defaultdict(int) for _ in range(n_envs)]
        self._ep_agent_rewards = [defaultdict(float) for _ in range(n_envs)]
        self._ep_food_remaining = [0] * n_envs

        # Prepare CSV
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_COLUMNS)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [{}] * len(rewards))

        for i in range(len(rewards)):
            self._ep_rewards[i] += rewards[i]
            self._ep_lengths[i] += 1

            # Extract info metrics from the environment
            info = infos[i] if i < len(infos) else {}
            self._ep_foods_collected[i] += info.get("foods_collected", 0)
            self._ep_coop_collections[i] += info.get("cooperative_collections", 0)
            self._ep_solo_collections[i] += info.get("solo_collections", 0)
            self._ep_failed_loads[i] += info.get("failed_loads", 0)
            self._ep_collisions[i] += info.get("collisions", 0)
            self._ep_food_remaining[i] = info.get("food_remaining", 0)

            # Accumulate action counts
            for action_name, count in info.get("action_counts", {}).items():
                self._ep_action_counts[i][action_name] += count

            # Accumulate per-agent rewards
            for j, r in enumerate(info.get("per_agent_rewards", [])):
                self._ep_agent_rewards[i][f"agent_{j}"] += r

            if dones[i]:
                self._ep_count += 1
                self._write_episode(i)
                self._reset_slot(i)

        return True

    def _write_episode(self, slot: int):
        """Write one finished episode to CSV and TensorBoard."""
        ep_reward = self._ep_rewards[slot]
        ep_length = self._ep_lengths[slot]
        foods = self._ep_foods_collected[slot]
        coop = self._ep_coop_collections[slot]
        solo = self._ep_solo_collections[slot]
        failed = self._ep_failed_loads[slot]
        collisions = self._ep_collisions[slot]
        food_remaining = self._ep_food_remaining[slot]
        ac = self._ep_action_counts[slot]
        agent_rewards = dict(self._ep_agent_rewards[slot])

        # Derived metrics
        coop_rate = coop / max(foods, 1)
        efficiency = foods / max(ep_length, 1)

        # -- TensorBoard --
        self.logger.record("episode/reward", ep_reward)
        self.logger.record("episode/length", ep_length)
        self.logger.record("episode/foods_collected", foods)
        self.logger.record("episode/cooperation_rate", coop_rate)
        self.logger.record("episode/collisions", collisions)
        self.logger.record("episode/food_remaining", food_remaining)
        self.logger.record("episode/collection_efficiency", efficiency)
        self.logger.record("episode/failed_loads", failed)
        self.logger.record("episode/count", self._ep_count)

        for agent_name, r in agent_rewards.items():
            self.logger.record(f"agents/{agent_name}_reward", r)

        # -- CSV --
        if self._csv_writer:
            self._csv_writer.writerow(
                [
                    self._ep_count,
                    self.num_timesteps,
                    time.time(),
                    round(ep_reward, 6),
                    ep_length,
                    foods,
                    coop,
                    solo,
                    failed,
                    food_remaining,
                    collisions,
                    ac.get("NORTH", 0),
                    ac.get("SOUTH", 0),
                    ac.get("EAST", 0),
                    ac.get("WEST", 0),
                    ac.get("LOAD", 0),
                    ac.get("NONE", 0),
                    json.dumps(agent_rewards),
                ]
            )
            if self._csv_file is not None:
                self._csv_file.flush()

    def _reset_slot(self, slot: int):
        """Reset accumulators for one vec-env slot."""
        self._ep_rewards[slot] = 0.0
        self._ep_lengths[slot] = 0
        self._ep_foods_collected[slot] = 0
        self._ep_coop_collections[slot] = 0
        self._ep_solo_collections[slot] = 0
        self._ep_failed_loads[slot] = 0
        self._ep_collisions[slot] = 0
        self._ep_action_counts[slot] = defaultdict(int)
        self._ep_agent_rewards[slot] = defaultdict(float)
        self._ep_food_remaining[slot] = 0

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
            print(f"  Metrics saved to: {self.csv_path}")


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
DEFAULT_PRESET = "fair"


def make_env(config=None, num_envs=1):
    """Create and wrap the foraging environment for SB3 training."""
    cfg = config or get_preset(DEFAULT_PRESET)
    env = AECForagingEnv(**cfg)
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    total_timesteps: int = 200_000,
    run_name: str | None = None,
    lr: float = 1e-3,
    preset: str = DEFAULT_PRESET,
    num_envs: int = 1,
    batch_size: int = 256,
):
    if run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")

    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = log_dir / "tb"
    csv_path = log_dir / "metrics.csv"
    model_path = log_dir / "model"
    config_path = log_dir / "config.json"

    env_config = get_preset(preset)

    # Save experiment config for reproducibility
    experiment_config = {
        "benchmark": BENCHMARK_NAME,
        "preset": preset,
        "run_name": run_name,
        "total_timesteps": total_timesteps,
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_envs": num_envs,
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "environment": env_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Convert tuples to lists for JSON serialization
    serializable_config = json.loads(
        json.dumps(experiment_config, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    )
    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=2)

    print(f"Run name   : {run_name}")
    print(f"Benchmark  : {BENCHMARK_NAME}")
    print(f"Preset     : {preset}")
    print(f"Log dir    : {log_dir}")
    print(f"Timesteps  : {total_timesteps:,}")
    print(f"LR         : {lr}")
    print(f"Num Envs   : {num_envs}")
    print(f"Batch Size : {batch_size}")
    print(f"Config     : {config_path}")
    print()

    # 1. Create environment
    env = make_env(env_config, num_envs=num_envs)

    # 2. Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr,
        batch_size=batch_size,
        tensorboard_log=str(tb_dir),
    )

    # 3. Create callback
    metrics_cb = MetricsCallback(csv_path=str(csv_path))

    # 4. Train
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=metrics_cb)

    # 5. Save model
    model.save(str(model_path))
    print(f"  Model saved to: {model_path}.zip")

    # 6. Quick evaluation
    print("\nEvaluating trained model (10 episodes)...")
    del env
    env = make_env(env_config)

    obs = env.reset()
    eval_rewards = []
    ep_reward = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = env.step(action)  # type: ignore
        ep_reward += sum(reward)
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
        print(f"  Min / Max      : {min(eval_rewards):.4f} / {max(eval_rewards):.4f}")

    env.close()
    print(f"\nDone! Visualize with:  uv run python -m scripts.visualize_logs {log_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on LB-Foraging")
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Run name (default: auto-generated timestamp)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list_presets(),
        help="Sustainable benchmark preset (default: fair)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available sustainable benchmark presets and exit",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of vectorized environments to run in parallel (default: 1). Increase this to speed up training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for PPO updates (default: 256).",
    )
    args = parser.parse_args()

    if args.list_presets:
        print(f"Benchmark: {BENCHMARK_NAME}")
        for preset_name in list_presets():
            print(f"- {preset_name}: {get_preset(preset_name)}")
        raise SystemExit(0)

    train(
        total_timesteps=args.timesteps,
        run_name=args.name,
        lr=args.lr,
        preset=args.preset,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
    )
