from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from scripts._bench_utils import MetricsTracker, save_experiment_config
from scripts.core.env_utils import make_env

class BenchmarkCallbacks(DefaultCallbacks):
    """Write episode stats to RLlib's custom metrics for driver-side logging."""
    def on_episode_end(self, *, episode: Any, env_index: int, **kwargs: Any) -> None:
        try:
            last_step_infos = dict(episode.get_infos(indices=-1))
        except Exception:
            all_infos = episode.get_infos()
            if not all_infos: return
            last_step_infos = all_infos[-1]

        if not last_step_infos: return
        first_agent = list(last_step_infos.keys())[0]
        last_info = last_step_infos[first_agent]

        if "episode_metrics" in last_info:
            metrics = last_info["episode_metrics"]
            for k, v in metrics.items():
                episode.custom_metrics[k] = v

def env_creator(config: dict[str, Any]) -> PettingZooEnv:
    env, _ = make_env(
        preset=config.get("preset", "fair"),
        num_envs=1,
        vectorize_for_cleanrl_sb3=False,
    )
    import supersuit as ss
    env = ss.flatten_v0(env)
    return PettingZooEnv(env)

def run_rllib(args: argparse.Namespace) -> None:
    run_name = args.name or f"rllib_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker = MetricsTracker(log_dir / "metrics.csv")

    save_experiment_config(
        log_dir, run_name, args.preset, algorithm="PPO", library="RLlib",
        total_timesteps=args.timesteps, learning_rate=args.lr, num_envs=args.num_envs,
        batch_size=args.batch_size, gamma=args.gamma,
    )

    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except Exception:
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count() or 1)

    env_name = "sustainable_foraging"
    register_env(env_name, env_creator)

    dummy_env = env_creator({"preset": args.preset})
    first_agent = list(dummy_env.get_agent_ids())[0]
    obs_space = getattr(dummy_env.observation_space, "spaces", {})[first_agent]
    act_space = getattr(dummy_env.action_space, "spaces", {})[first_agent]
    policy_mapping_fn = lambda agent_id, *_args, **_kwargs: "shared_policy"

    config = (
        PPOConfig()
        .environment(env_name, env_config={"preset": args.preset})
        .env_runners(num_env_runners=max(1, args.num_envs - 1))
        .training(train_batch_size=args.batch_size, lr=args.lr, gamma=args.gamma)
        .multi_agent(policies={"shared_policy": (None, obs_space, act_space, {})}, policy_mapping_fn=policy_mapping_fn)
        .callbacks(BenchmarkCallbacks)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    algo = config.build()

    total_steps = 0
    while total_steps < args.timesteps:
        result = algo.train()
        total_steps = int(result.get("timesteps_total", total_steps))
        cm = result.get("custom_metrics", {})
        
        if cm and "reward_total_mean" in cm:
            stats = {
                "reward_total": cm.get("reward_total_mean", 0),
                "length": cm.get("length_mean", 0),
                "foods_collected": cm.get("foods_collected_mean", 0),
                "cooperative_collections": cm.get("cooperative_collections_mean", 0),
                "solo_collections": cm.get("solo_collections_mean", 0),
                "failed_loads": cm.get("failed_loads_mean", 0),
                "food_remaining_end": cm.get("food_remaining_end_mean", 0),
                "collisions": cm.get("collisions_mean", 0),
                "action_counts": {}, "agent_rewards": {},
            }
            tracker.on_episode_end(total_steps, stats)

        print(f"  Steps: {total_steps:>8,}/{args.timesteps:,} | Reward: {result.get('episode_reward_mean', 0.0):>7.2f}")

    algo.save(str(log_dir / "model"))
    if tracker: tracker.close()
    ray.shutdown()
