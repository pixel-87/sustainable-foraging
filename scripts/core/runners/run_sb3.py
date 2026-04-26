from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

from scripts._bench_utils import MetricsTracker, save_experiment_config
from scripts.core.env_utils import make_env
from sustainable_foraging.foraging.sustainable_benchmark import BENCHMARK_NAME


class EpisodeMetricsCallback(BaseCallback):
    """Extracts episode metrics attached to info by ForagingMetricsWrapper."""
    def __init__(self, tracker: MetricsTracker, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._tracker = tracker

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_metrics" in info:
                metrics = info["episode_metrics"]
                self._tracker.on_episode_end(self.num_timesteps, metrics)
                
                # Also log to TensorBoard
                self.logger.record("episode/reward", metrics["reward_total"])
                self.logger.record("episode/length", metrics["length"])
                self.logger.record("episode/foods_collected", metrics["foods_collected"])
                self.logger.record("episode/count", self._tracker.episode_count)
        return True

    def _on_training_end(self) -> None:
        self._tracker.close()


def run_sb3(args: argparse.Namespace, algorithm: str = "ppo") -> None:
    algo_name = algorithm.upper()
    run_name = args.name or f"sb3_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = log_dir / "tb"
    model_path = log_dir / "model"
    csv_path = log_dir / "metrics.csv"

    # Create env
    env, env_config = make_env(
        preset=args.preset,
        num_envs=args.num_envs,
        vectorize_for_cleanrl_sb3=True,
        base_class="stable_baselines3"
    )

    # Save config
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm=algo_name,
        library="SB3",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        policy="MlpPolicy",
    )

    print(f"Run name   : {run_name}")
    print(f"Benchmark  : {BENCHMARK_NAME}")
    print(f"Preset     : {args.preset}")
    print(f"Log dir    : {log_dir}")
    print(f"Timesteps  : {args.timesteps:,}")
    print(f"LR         : {args.lr}")
    print(f"Num Envs   : {args.num_envs}")
    print(f"Batch Size : {args.batch_size}")
    print()

    # Create model
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tensorboard_log=str(tb_dir),
        )
    elif algo_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            gamma=args.gamma,
            tensorboard_log=str(tb_dir),
        )
    else:
        raise ValueError(f"Unknown SB3 algorithm: {algo_name}")

    # Train
    tracker = MetricsTracker(csv_path)
    callback = EpisodeMetricsCallback(tracker)
    print(f"Starting {algo_name} training...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save
    model.save(str(model_path))
    print(f"  Model saved to: {model_path}.zip")

    # Evaluate
    print("\nEvaluating trained model (10 episodes)...")
    env.close()
    eval_env, _ = make_env(preset=args.preset, num_envs=1, vectorize_for_cleanrl_sb3=True, base_class="stable_baselines3")
    
    obs = eval_env.reset()
    eval_rewards = []
    ep_reward = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        ep_reward += float(sum(reward))
        if any(done):
            eval_rewards.append(ep_reward)
            ep_reward = 0.0
            obs = eval_env.reset()
            if len(eval_rewards) >= 10:
                break
                
    if eval_rewards:
        print(f"  Eval episodes  : {len(eval_rewards)}")
        print(f"  Mean reward    : {np.mean(eval_rewards):.4f}")
        print(f"  Std reward     : {np.std(eval_rewards):.4f}")

    eval_env.close()
    print(f"\nDone! Visualize with:  uv run python -m scripts.compare_algorithms logs/{run_name}")
