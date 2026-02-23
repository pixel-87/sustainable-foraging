import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo.utils import aec_to_parallel
from lbforaging.foraging import AECForagingEnv
import time

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained PPO model")
    parser.add_argument(
        "--model", "-m", type=str, default="ppo_lbforaging",
        help="Path to the trained model zip file (without .zip extension)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum number of steps per episode (default: 10000)"
    )
    args = parser.parse_args()

    print("Initializing environment...")
    # 1. Re-create the environment with the EXACT same parameters as training
    # We add render_mode="human" to enable visual output
    env = AECForagingEnv(
        players=2,
        max_energy=100,
        food_energy_value=10,
        energy_depletion_rate=1,
        food_regeneration_rate=0.1,
        num_food_zones=2,
        field_size=(8, 8),
        max_num_food=2,
        sight=8,
        max_episode_steps=args.max_steps,
        render_mode="human",
        grid_observation=True
    )

    # 2. Apply the EXACT same wrapper stack as used in training
    # This ensures observations and actions match the trained policy
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')

    # 3. Load the model
    print(f"Loading model '{args.model}.zip'...")
    try:
        model = PPO.load(args.model)
    except FileNotFoundError:
        print(f"Error: Could not find '{args.model}.zip'. Make sure you have run train_sb3.py first.")
        return

    # 4. Inference Loop
    print("\nStarting inference loop...")
    print("Press Ctrl+C to stop.")
    
    obs = env.reset()
    total_reward = 0
    episodes = 0
    
    try:
        while True:
            # predict returns (action, state)
            # deterministic=True usually gives better performance for evaluation
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = env.step(action)
            
            # Aggregate reward for display
            total_reward += np.sum(reward)
            
            env.render()
            
            # Control framerate slightly so it's not too fast
            time.sleep(0.05)
            
            # Check if any environment in the vector is done
            if any(done):
                episodes += 1
                print(f"Episode {episodes} finished. Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
