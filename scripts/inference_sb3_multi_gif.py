#!/usr/bin/env python3
"""Run inference with a pre‑trained PPO model for several episodes and save a GIF.

The script loads an existing ``*.zip`` checkpoint (produced by ``scripts/train_sb3.py`` or any
compatible PPO model) and records *multiple* episodes, resetting the environment between them.
It is useful for a quick visual demo of agents moving around the foraging grid without any
additional training.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import imageio
import supersuit as ss
from stable_baselines3 import PPO
from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import (
    BENCHMARK_NAME,
    get_preset,
    list_presets,
)
from pettingzoo.utils import aec_to_parallel

DEFAULT_PRESET = "fair"
DEFAULT_EPISODES = 3
DEFAULT_MAX_STEPS = 1000
DEFAULT_FPS = 15
DEFAULT_OUTPUT = "multi_episode_demo.gif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record several episodes of a pre‑trained PPO model and save them as a GIF"
    )
    p.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Base name (without .zip) of the trained PPO model",
    )
    p.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list_presets(),
        help="Benchmark preset (default: fair)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="How many episodes to record (default: 3)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum steps per episode (default: 1000)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second for the GIF (default: 15)",
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

    # Load the model
    model_path = f"{args.model}.zip"
    if not Path(model_path).exists():
        print(f"Error: model file {model_path} not found. Use the correct base name.")
        return
    print(f"Loading PPO model from {model_path} …")
    model = PPO.load(args.model)

    # Prepare environment
    env = make_env(args.preset, args.max_steps)

    all_frames: list[np.ndarray] = []
    for ep in range(1, args.episodes + 1):
        print(f"Recording episode {ep}/{args.episodes} …")
        obs = env.reset()
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                all_frames.append(frame)
            else:
                all_frames.append(frame[0])
            step += 1
            time.sleep(1.0 / args.fps)
            if any(done) or step >= args.max_steps:
                print(f"  Episode finished after {step} steps.")
                # Insert a short pause (blank frame) between episodes for visual separation
                blank = np.zeros_like(all_frames[-1])
                for _ in range(int(args.fps * 0.5)):  # half‑second pause
                    all_frames.append(blank)
                break
    env.close()

    # Save GIF
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_uint8 = [np.asarray(f, dtype=np.uint8) for f in all_frames]
    print(f"Saving GIF to {out_path} …")
    imageio.mimsave(out_path, frames_uint8, fps=args.fps)
    print("GIF saved – enjoy!")


if __name__ == "__main__":
    main()
