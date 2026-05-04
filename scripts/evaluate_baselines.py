#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from core.env_utils import make_env
from sustainable_foraging.foraging.aecEnvironment import Action

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def random_policy(obs, env, agent):
    """Returns a random action from the agent's action space."""
    return env.action_space(agent).sample()

def greedy_policy(obs, env, agent):
    """
    Myopic baseline: Moves towards the closest visible food.
    If adjacent to food, loads. If no food visible, moves randomly.
    Acts purely on the raw observation array (NumPy).
    Handles both flat (1D) and grid (3D) observations.
    """
    if len(obs.shape) == 3:
        # Grid observation: shape is (3, H, W)
        # Layer 0: agents, Layer 1: foods, Layer 2: access
        player_y, player_x = obs.shape[1] // 2, obs.shape[2] // 2
        
        foods_y, foods_x = np.nonzero(obs[1])
        if len(foods_y) == 0:
            return env.action_space(agent).sample()
            
        min_dist = float('inf')
        closest_food = None
        
        for fy, fx in zip(foods_y, foods_x):
            dist = abs(fy - player_y) + abs(fx - player_x)
            if dist < min_dist:
                min_dist = dist
                closest_food = (fy, fx)
                
        if closest_food is None:
            return env.action_space(agent).sample()
            
        fy, fx = closest_food
        
    else:
        # Flat observation
        max_num_food = env.unwrapped.max_num_food
        player_idx = max_num_food * 2
        player_y = obs[player_idx]
        player_x = obs[player_idx + 1]

        if player_y == -1 or player_x == -1:
            return env.action_space(agent).sample()

        min_dist = float('inf')
        closest_food = None

        for i in range(max_num_food):
            fy, fx = obs[2*i], obs[2*i+1]
            if fy == -1 and fx == -1:
                continue
            dist = abs(fy - player_y) + abs(fx - player_x)
            if dist < min_dist:
                min_dist = dist
                closest_food = (fy, fx)

        if closest_food is None:
            return env.action_space(agent).sample()

        fy, fx = closest_food

    if min_dist == 1:
        return 5  # LOAD

    # NORTH = 1 (y - 1), SOUTH = 2 (y + 1), WEST = 3 (x - 1), EAST = 4 (x + 1)
    if fy < player_y:
        return 1
    elif fy > player_y:
        return 2
    elif fx < player_x:
        return 3
    elif fx > player_x:
        return 4

    return env.action_space(agent).sample()

def evaluate(policy_fn, policy_name, num_episodes=500, sight=2, seed=1):
    # Initialize the base AEC environment with the fair preset
    from sustainable_foraging.foraging import AECForagingEnv
    from sustainable_foraging.foraging.sustainable_benchmark import get_preset
    from core.log_utils import ForagingMetricsWrapper
    
    env_config = get_preset("fair")
    env_config["sight"] = sight
    base_env = AECForagingEnv(**env_config)
    env = ForagingMetricsWrapper(base_env)

    logger.info(f"Evaluating {policy_name} (sight={sight}, seed={seed}) for {num_episodes} episodes...")

    metrics = []

    env.reset(seed=seed)
    for ep in tqdm(range(num_episodes), desc=f"{policy_name} s={sight} seed={seed}"):
        if ep > 0:
            env.reset()
        
        while getattr(env, "agents", None):
            agent = env.agent_selection
            obs, reward, terminated, truncated, info = env.last()
            
            if terminated or truncated:
                action = None
            else:
                action = policy_fn(obs, env, agent)
                
            env.step(action)
            
            # Check if this agent finished the episode and recorded metrics
            if (terminated or truncated) and "episode_metrics" in info:
                if len(metrics) == 0 or info["episode_metrics"] != metrics[-1]:
                    metrics.append(info["episode_metrics"])
                    
    # Only keep exactly num_episodes
    metrics = metrics[:num_episodes]

    # Calculate averages
    avg_reward = np.mean([m["reward_total"] for m in metrics])
    avg_length = np.mean([m["length"] for m in metrics])
    avg_foods = np.mean([m["foods_collected"] for m in metrics])
    avg_sustainability = np.mean([m["food_remaining_end"] for m in metrics])

    logger.info(f"Results for {policy_name} (sight={sight}, seed={seed}):")
    logger.info(f"  Avg Reward: {avg_reward:.2f}")
    logger.info(f"  Avg Length: {avg_length:.2f}")
    logger.info(f"  Avg Foods:  {avg_foods:.2f}")
    logger.info(f"  Avg Sustain:{avg_sustainability:.2f}")

    return metrics

def write_logs(policy_name, metrics, obs_type="pomdp", seed=1, total_timesteps=5_000_000):
    """
    Writes the metrics to CSV logs compatible with compare_algorithms.py.
    We linearly interpolate the episode timesteps across `total_timesteps`
    so the baselines appear as continuous flat lines on the graphs.
    """
    import os
    run_dir = Path(f"logs/baseline_{policy_name.lower()}_{obs_type}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config.json
    config = {
        "library": "Baseline",
        "algorithm": policy_name
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Write metrics.csv
    num_points = 100
    timesteps = np.linspace(0, total_timesteps, num_points, dtype=int)
    
    # Calculate average metrics over all episodes
    avg_reward = np.mean([m["reward_total"] for m in metrics])
    avg_length = np.mean([m["length"] for m in metrics])
    avg_foods = np.mean([m["foods_collected"] for m in metrics])
    avg_sustainability = np.mean([m["food_remaining_end"] for m in metrics])
    
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "reward_total", "length", "foods_collected", "food_remaining_end"])
        
        for ts in timesteps:
            writer.writerow([
                int(ts),
                avg_reward,
                avg_length,
                avg_foods,
                avg_sustainability
            ])
            
    logger.info(f"Saved {policy_name} logs to {run_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    for sight, obs_type in [(2, "pomdp"), (8, "fomdp")]:
        for seed in [1, 2, 3]:
            random_metrics = evaluate(random_policy, "Random", args.episodes, sight=sight, seed=seed)
            write_logs("Random", random_metrics, obs_type=obs_type, seed=seed)

            greedy_metrics = evaluate(greedy_policy, "Greedy", args.episodes, sight=sight, seed=seed)
            write_logs("Greedy", greedy_metrics, obs_type=obs_type, seed=seed)

if __name__ == "__main__":
    main()
