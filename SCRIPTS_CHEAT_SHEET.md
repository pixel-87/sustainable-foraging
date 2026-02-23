# LB-Foraging Scripts & Commands Cheat Sheet

### **Run Tests**
**Command:** `pytest tests/test_aec.py`
*   **Purpose:** Runs the test suite to verify the AEC environment logic (rewards, spawning, observability).
*   **Key Tests:**
    *   `test_aec_api`: Checks strict PettingZoo API compliance.
    *   `test_seeding_reproducibility`: Ensures consistent runs with same seed.
    *   `test_reward_cooperative_loading`: Verifies shared rewards.

### **Interactive Play (Original Env)**
**Command:** `python -m scripts.human_play`
*   **Purpose:** Play the game manually to understand the mechanics.
*   **Options:**
    *   `--env Foraging-8x8-2p-2f-v3`: Select specific map configuration.
    *   `--max_steps 100`: Set episode length.
    *   `--display_info`: Show debug info in terminal.
*   **Controls:** Arrow keys to move, `L` to load food, `TAB` to switch agents.


### **Train a New Model**
**Command:** `python -m scripts.train_sb3 --preset fair`
*   **Purpose:** Trains a PPO agent from scratch using the AEC environment.
*   **Output:** Saves logs/config/model under `logs/<run_name>/`.
*   **Preset tiers:**
    *   `easy`: High regeneration, generous energy budget.
    *   `fair`: Balanced benchmark default.
    *   `hard`: Low regeneration, high energy pressure.
*   **Tip:** `python -m scripts.train_sb3 --list-presets`

### **Visual Inference (Run Saved Model)**
**Command:** `python -m scripts.inference_sb3 --preset fair --model logs/<run_name>/model`
*   **Purpose:** Watch the trained agents play in real-time.
*   **Requirement:** Use the same preset as training.
*   **Key Features:**
    *   `render_mode="human"`: Opens a window to show the game.
    *   `deterministic=True`: Uses the best predicted action (no random exploration).


### **Environment Versions**
*   **`GymForagingEnv`** (Original): The standard Gym environment. Used by `scripts.human_play`. Agents often have hardcoded heuristic controllers (`H1-H4`).
*   **`AECForagingEnv`** (New): The PettingZoo AEC implementation. Used by `scripts.train_sb3`. Designed for multi-agent RL from scratch.

### **Policy Types**
*   **`MlpPolicy`**: Simple "feed-forward" network. Treats the board as a flat list of numbers. Good for simple states, bad for grids. *Current active config.*
*   **`CnnPolicy`**: "Convolutional" network. Treats the board as an image. Essential for agents to "see" spatial patterns.

### **Observation Modes**
*   **`grid_observation=False`**: Returns a flat vector (Player positions X,Y + Food positions X,Y).
*   **`grid_observation=True`**: Returns a 3D Tensor (Layers x Height x Width). Normalized to `[0, 1]`. *Current active config.* 

## 4. Random Agent Testing
Use this to check that the environment loop works without any training intelligence (random actions).

### **Run Random Agents (Gym Version)**
**Command:** `python -m scripts.lbforaging --render --episodes 5`
*   **Purpose:** Runs the *original* Gym environment with random actions.
*   **Output:** Prints returns and renders the game.

### **Run Random Agents (AEC Version)**
**Command:** `python -m scripts.lbforaging --render --aec --episodes 5`
*   **Purpose:** Runs the *new PettingZoo AEC* environment with random actions.
*   **Why use this?**
    *   Verifies that your AEC `step()` and `render()` logic doesn't crash.
    *   Shows the visual output of the new environment code without needing to train a model first.
