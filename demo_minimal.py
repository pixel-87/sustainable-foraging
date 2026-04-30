import gymnasium as gym
import numpy as np
from sustainable_foraging.foraging import AECForagingEnv
from stable_baselines3 import PPO as SB3_PPO
from supersuit import pettingzoo_env_to_vec_env_v1
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.utils.conversions import aec_to_parallel
import time

# 1. Initialize a minimal sustainable environment
env = AECForagingEnv(players=3, field_size=(10, 10), render_mode="human")

# 2. Convert to parallel
env = aec_to_parallel(env)

# 3. Add necessary wrappers
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)

# 4. Convert to vectorized environment and then to SB3 format
env = pettingzoo_env_to_vec_env_v1(env)
env = SB3VecEnvWrapper(env)

# 5. Plug and Play: Stable Baselines3 (PPO)
model = SB3_PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=5000)

print("Training complete. Starting visualization demo...")

# Visualization loop
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
    time.sleep(0.1)
    if np.any(dones):
        obs = env.reset()

print("Demo complete: Visualization finished.")
