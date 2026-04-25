#!/usr/bin/env python3
"""Run inference on a trained PPO model and save a GIF of the episode.

This script mirrors ``scripts/inference_sb3.py`` but captures each rendered frame
as an RGB array and writes the collected frames to a GIF file using ``imageio``.
It is useful for creating visualisations for presentations or papers.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import supersuit as ss
import imageio
from lbforaging.foraging import AECForagingEnv
from lbforaging.foraging.sustainable_benchmark import (
    BENCHMARK_NAME,
    get_preset,
    list_presets,
)
from pettingzoo.utils import aec_to_parallel
from stable_baselines3 import PPO

DEFAULT_PRESET = "fair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a PPO model and save a GIF")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="ppo_lbforaging",
        help="Base name of the trained model zip file (without .zip extension)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list_presets(),
        help="Sustainable benchmark preset (default: fair)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output GIF (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="episode.gif",
        help="Path to the output GIF file",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available sustainable benchmark presets and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_presets:
        print(f"Benchmark: {BENCHMARK_NAME}")
        for name in list_presets():
            print(f"- {name}: {get_preset(name)}")
        return

    # Load environment configuration
    env_cfg = get_preset(args.preset)
    env_cfg["max_episode_steps"] = args.max_steps

    # Create the environment with render_mode="rgb_array" to capture frames
    env = AECForagingEnv(render_mode="rgb_array", **env_cfg)
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

    # Load the PPO model
    model_path = f"{args.model}.zip"
    if not Path(model_path).exists():
        print(f"Error: model file {model_path} not found. Run scripts/train_sb3.py first.")
        return
    print(f"Loading model {model_path} ...")
    model = PPO.load(args.model)

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    obs = env.reset()
    print("Starting inference and recording frames …")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # Capture frame
            frame = env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            else:
                frames.append(frame[0])
            time.sleep(1.0 / args.fps)
            if any(done):
                print(f"Episode finished after {len(frames)} frames.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted – writing captured frames.")
    finally:
        env.close()

    if frames:
        frames_uint8 = [np.asarray(f, dtype=np.uint8) for f in frames]
        print(f"Saving GIF to {output_path} …")
        imageio.mimsave(output_path, frames_uint8, fps=args.fps)
        print("GIF saved successfully.")
    else:
        print("No frames captured – GIF not created.")


if __name__ == "__main__":
    main()
