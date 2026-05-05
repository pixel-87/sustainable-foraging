<<<<<<< HEAD
from sustainable_foraging.foraging import AECForagingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from supersuit import pettingzoo_env_to_vec_env_v1
import supersuit as ss
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from pettingzoo.utils.conversions import aec_to_parallel
import numpy as np

class RenderCallback(BaseCallback):
    """Callback for rendering the environment during training to show progress."""
=======
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from supersuit import pettingzoo_env_to_vec_env_v1
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from sustainable_foraging.foraging import AECForagingEnv


class RenderCallback(BaseCallback):
    """Callback for rendering the environment during training to show progress."""

>>>>>>> cleanup-legacy-files
    def _on_step(self) -> bool:
        self.training_env.render()
        return True

<<<<<<< HEAD
=======

>>>>>>> cleanup-legacy-files
# 1. Initialize environment with human rendering
env = AECForagingEnv(players=3, field_size=(12, 12), render_mode="human")

# 2. Convert to parallel
env = aec_to_parallel(env)

# 3. Add necessary wrappers
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)

# 4. Convert to vectorized environment and then to SB3 format
env = pettingzoo_env_to_vec_env_v1(env)
env = SB3VecEnvWrapper(env)

# 5. Off-policy Learning: DQN
# Using a high exploration rate initially so people see the "learning" process
model = DQN(
<<<<<<< HEAD
    "MlpPolicy", 
    env, 
    verbose=1, 
=======
    "MlpPolicy",
    env,
    verbose=1,
>>>>>>> cleanup-legacy-files
    device="cpu",
    buffer_size=50_000,
    learning_starts=500,
    batch_size=64,
    gamma=0.99,
<<<<<<< HEAD
    exploration_fraction=0.5, # Takes longer to become greedy, showing the transition
=======
    exploration_fraction=0.5,  # Takes longer to become greedy, showing the transition
>>>>>>> cleanup-legacy-files
    exploration_final_eps=0.05,
)

print("Starting continuous visual training (Ctrl+C to stop)...")
print("The agents will start with random moves and slowly improve their foraging.")

try:
    # Train indefinitely while rendering every single step
    model.learn(total_timesteps=int(1e9), callback=RenderCallback())

except KeyboardInterrupt:
    print("\nVisual demo stopped.")
finally:
    env.close()
