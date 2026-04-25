#!/usr/bin/env python3
"""Train a PPO agent using Ray RLlib on the Sustainable Foraging AEC environment.

RLlib natively supports PettingZoo AEC environments via ``PettingZooEnv``.
This script uses parameter sharing so all agents learn from a single policy.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import get_preset
from scripts._bench_utils import MetricsTracker, get_standard_parser, save_experiment_config

# ---------------------------------------------------------------------------
# Environment creator
# ---------------------------------------------------------------------------


def env_creator(config: dict[str, Any]) -> PettingZooEnv:
    """Instantiate the PettingZoo AEC environment wrapped for RLlib."""
    print("DEBUG: env_creator called")
    env_config: dict[str, Any] = get_preset(config.get("preset", "fair"))
    env = AECForagingEnv(**env_config)
    print("DEBUG: AECForagingEnv instantiated")
    
    import supersuit as ss
    # Flatten the (3, 17, 17) grid into a 1D vector so RLlib doesn't try
    # (and fail) to build a default CNN for this non-standard shape.
    env = ss.flatten_v0(env)
    print("DEBUG: env flattened with ss")
    pz_env = PettingZooEnv(env)
    print("DEBUG: PettingZooEnv wrapper created")
    return pz_env



# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class BenchmarkCallbacks(DefaultCallbacks):
    """Write episode stats to RLlib's custom metrics for driver-side logging."""

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_index: int,
        **kwargs: Any,
    ) -> None:
        # In the new API stack, episode is a MultiAgentEpisode.
        # We can fetch the last info dict (which maps agent_id -> info)
        try:
            last_step_infos = dict(episode.get_infos(indices=-1))
        except Exception:
            # Fallback if indices=-1 fails or isn't supported in this RLlib version
            all_infos = episode.get_infos()
            if not all_infos:
                return
            last_step_infos = all_infos[-1]

        if not last_step_infos:
            return

        # They all share the same global statistics, so we can just grab the first one
        first_agent = list(last_step_infos.keys())[0]
        last_info = last_step_infos[first_agent]

        total_reward: float = sum(episode.get_return().values())

        stats: dict[str, Any] = {
            "reward_total": total_reward,
            "length": episode.env_t,
            "foods_collected": last_info.get("foods_collected", 0),
            "cooperative_collections": last_info.get("cooperative_collections", 0),
            "solo_collections": last_info.get("solo_collections", 0),
            "failed_loads": last_info.get("failed_loads", 0),
            "food_remaining_end": last_info.get("food_remaining", 0),
            "collisions": last_info.get("collisions", 0),
        }

        # Store these directly in the custom metrics so RLlib aggregates them
        for k, v in stats.items():
            episode.custom_metrics[k] = v


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    run_name: str = args.name or f"rllib_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir: Path = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    tracker = MetricsTracker(log_dir / "metrics.csv")

    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="PPO",
        library="RLlib",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    # Initialize Ray. Try to connect to an existing cluster first (e.g. if the user
    # ran `ray start --head` to avoid GCS timeouts on NixOS).
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except Exception:
        ray.init(
            ignore_reinit_error=True, 
            num_cpus=os.cpu_count() or 1,
        )

    env_name: str = "sustainable_foraging"
    register_env(env_name, env_creator)

    # Inspect spaces from a throwaway env
    dummy_env: PettingZooEnv = env_creator({"preset": args.preset})
    # dummy_env.observation_space is a Dict mapping agent_id -> Box
    # For parameter sharing, we just want the base space for one agent
    first_agent = list(dummy_env.get_agent_ids())[0]
    obs_space = getattr(dummy_env.observation_space, "spaces", {})[first_agent]
    act_space = getattr(dummy_env.action_space, "spaces", {})[first_agent]

    # Parameter sharing: one policy for every agent
    policy_mapping_fn = lambda agent_id, *_args, **_kwargs: "shared_policy"

    config = (
        PPOConfig()
        .environment(env_name, env_config={"preset": args.preset})
        .env_runners(num_env_runners=max(1, args.num_envs - 1))
        .training(
            train_batch_size=args.batch_size,
            lr=args.lr,
            gamma=args.gamma,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .callbacks(BenchmarkCallbacks)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    algo = config.build()

    print(f"Training RLlib PPO | {args.timesteps:,} steps | preset={args.preset}")

    total_steps: int = 0
    while total_steps < args.timesteps:
        result: dict[str, Any] = algo.train()
        total_steps = int(result.get("timesteps_total", total_steps))
        
        cm = result.get("custom_metrics", {})
        if cm:
            stats = {
                "reward_total": cm.get("reward_total_mean", 0),
                "length": cm.get("length_mean", 0),
                "foods_collected": cm.get("foods_collected_mean", 0),
                "cooperative_collections": cm.get("cooperative_collections_mean", 0),
                "solo_collections": cm.get("solo_collections_mean", 0),
                "failed_loads": cm.get("failed_loads_mean", 0),
                "food_remaining_end": cm.get("food_remaining_end_mean", 0),
                "collisions": cm.get("collisions_mean", 0),
                "action_counts": {},
                "agent_rewards": {},
            }
            tracker.on_episode_end(total_steps, stats)

        reward_mean: float = result.get("episode_reward_mean", 0.0)
        len_mean: float = result.get("episode_len_mean", 0.0)
        print(
            f"  Steps: {total_steps:>8,}/{args.timesteps:,} | "
            f"Reward: {reward_mean:>7.2f} | Len: {len_mean:>5.1f}"
        )

    algo.save(str(log_dir / "model"))
    print(f"Training complete. Model saved to {log_dir / 'model'}")
    print(f"Visualize with:  uv run python -m scripts.compare_algorithms logs/{run_name}")

    if tracker:
        tracker.close()
    ray.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_standard_parser(
        description="Train PPO (RLlib) on Sustainable Foraging"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    train(parser.parse_args())
