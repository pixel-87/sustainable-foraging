#!/usr/bin/env python3
"""Train a PPO agent for a short burst and record a GIF of the first episode.

The script is deliberately lightweight:
* 200 k training steps (configurable)
* records only the *first* episode that the agent plays after the first reset
* saves a high‑fps GIF (default 30 fps) so the motion looks lively
* uses the same environment‑wrapping stack as the regular training scripts
"""

import argparse
import time
from pathlib import Path

import numpy as np
import imageio
import supersuit as ss
from stable_baselines3 import PPO
from lbforaging.foraging import AECForagingEnv
from lbforaging.foraging.sustainable_benchmark import (
    BENCHMARK_NAME,
    get_preset,
    list_presets,
)
from pettingzoo.utils import aec_to_parallel

DEFAULT_PRESET = "fair"
DEFAULT_TIMESTEPS = 200_000
DEFAULT_FPS = 30
DEFAULT_OUTPUT = "training_demo.gif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO briefly and capture a GIF of the first episode"
    )
    p.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list_presets(),
        help="Benchmark preset (default: fair)",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Training steps (default: 200 000)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode during recording (default: 1000)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second for the output GIF (default: 30)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path for the generated GIF",
    )
    p.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available benchmark presets and exit",
    )
    return p.parse_args()


def make_env(preset: str, max_steps: int):
    cfg = get_preset(preset).copy()
    cfg["max_episode_steps"] = max_steps
    env = AECForagingEnv(render_mode="rgb_array", **cfg)
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")
    return env


def main() -> None:
    args = parse_args()

    if args.list_presets:
        print(f"Benchmark: {BENCHMARK_NAME}")
        for n in list_presets():
            print(f"- {n}: {get_preset(n)}")
        return

    # ----------------------------------------------------------------------
    # 1️⃣ Build the environment (same wrappers as training)
    # ----------------------------------------------------------------------
    env = make_env(args.preset, args.max_steps)

    # ----------------------------------------------------------------------
    # 2️⃣ Train a PPO model for a short burst
    # ----------------------------------------------------------------------
    print(f"Training PPO for {args.timesteps:,} steps (this will be quick)…")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=args.timesteps)

    # ----------------------------------------------------------------------
    # 3️⃣ Record the first episode after training
    # ----------------------------------------------------------------------
    print("Recording the first episode …")
    frames = []
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=False)  # stochastic for fun
        obs, reward, done, info = env.step(action)
        # Capture rendered frame
        frame = env.render(mode="rgb_array")
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        else:
            frames.append(frame[0])
        time.sleep(1.0 / args.fps)
        if any(done):
            print(f"Episode finished after {len(frames)} frames.")
            break

    env.close()

    # ----------------------------------------------------------------------
    # 4️⃣ Write the GIF
    # ----------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_uint8 = [np.asarray(f, dtype=np.uint8) for f in frames]
    print(f"Saving GIF to {out_path} …")
    imageio.mimsave(out_path, frames_uint8, fps=args.fps)
    print("GIF saved – enjoy!")


if __name__ == "__main__":
    main()
