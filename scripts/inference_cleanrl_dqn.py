#!/usr/bin/env python3
"""Run inference on a trained CleanRL DQN model with visualization.

This script loads a model.pt from a CleanRL DQN run and runs it in the
environment with render_mode="human" to show the grid.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn

from lbforaging.foraging import AECForagingEnv
from lbforaging.foraging.sustainable_benchmark import get_preset, list_presets
from pettingzoo.utils import aec_to_parallel

# ---------------------------------------------------------------------------
# Network (Must match train_cleanrl_dqn.py)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference on trained CleanRL DQN model")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the trained model.pt file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fair",
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
        help="Frames per second for visualization (default: 10)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Initializing environment with preset: {args.preset}...")
    env_config = get_preset(args.preset)
    env_config["max_episode_steps"] = args.max_steps

    # Create the environment with render_mode="human"
    env = AECForagingEnv(render_mode="human", **env_config)
    
    # Apply the same wrapper stack as used in training
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class="gymnasium")

    # Get dimensions
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    num_agents = env.num_envs

    # Load agent
    print(f"Loading model from {model_path}...")
    device = torch.device("cpu")
    q_network = QNetwork(obs_dim, act_dim).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    print("\nStarting visual inference...")
    print("Close the window or press Ctrl+C to stop.")

    try:
        obs, _ = env.reset()
        total_reward = 0
        episodes = 0
        
        while True:
            # Prepare observation for torch
            obs_torch = torch.tensor(obs, dtype=torch.float32, device=device).reshape(num_agents, -1)
            
            with torch.no_grad():
                # For DQN inference, we just take the argmax of Q-values
                q_values = q_network(obs_torch)
                action = q_values.argmax(dim=1).cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)
            
            total_reward += np.sum(reward)
            
            # Rendering
            env.render()
            
            time.sleep(1.0 / args.fps)

            if any(done):
                episodes += 1
                print(f"Episode {episodes} finished. Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nInference stopped.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
