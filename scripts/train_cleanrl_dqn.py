#!/usr/bin/env python3
"""Train Independent DQN on Sustainable Foraging using CleanRL-style code.

Single-file, dependency-light implementation following CleanRL conventions:
  - Pure PyTorch + gymnasium/PettingZoo
  - Parameter sharing: one Q-network for all homogeneous agents
  - Experience replay buffer
  - Epsilon-greedy exploration with linear decay
  - Target network with hard updates
  - Consistent CSV / config logging via ``_bench_utils``

Logs training metrics to:
  - CSV file                 (./logs/<run_name>/metrics.csv)
  - Experiment config        (./logs/<run_name>/config.json)
  - TensorBoard events       (./logs/<run_name>/tb/)
  - Saved model              (./logs/<run_name>/model.pt)
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lbforaging.foraging import AECForagingEnv
from lbforaging.foraging.sustainable_benchmark import get_preset, get_training_defaults
from pettingzoo.utils import aec_to_parallel
from scripts._bench_utils import MetricsTracker, get_standard_parser, save_experiment_config

# ---------------------------------------------------------------------------
# Hyperparameter defaults
# ---------------------------------------------------------------------------
TRAINING_DEFAULTS = get_training_defaults()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete action spaces."""

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
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Simple numpy-backed replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: int, reward: float, done: bool):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idxs], device=self.device),
            torch.as_tensor(self.next_obs[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device).long(),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device),
        )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(preset: str = "fair", num_envs: int = 1):
    """Create vectorized env using the same wrapping chain as SB3/MAPPO.

    Note: grid_observation is forced to False for DQN.  The 17x17x3 grid
    observation (867 dims) is prohibitively slow on CPU for a DQN replay
    buffer.  Flat vector observations (~30 dims) give the same information
    content and run ~20x faster through the MLP Q-network.
    """
    env_config = get_preset(preset)
    env_config["grid_observation"] = False  # Force flat obs for DQN speed
    env = AECForagingEnv(**env_config)
    env = aec_to_parallel(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=0,
        base_class="gymnasium",
    )
    return env, env_config



# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    """Linear epsilon decay."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train(args: argparse.Namespace) -> None:
    run_name: str = args.name or f"cleanrl_dqn_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # -- Seeding --
    seed = getattr(args, "seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu"
    )

    # -- Environment --
    env, env_config = make_env(args.preset, num_envs=args.num_envs)
    single_obs_space = env.observation_space
    single_act_space = env.action_space
    obs_dim = int(np.prod(single_obs_space.shape))
    act_dim = int(single_act_space.n)
    num_envs_total = env.num_envs

    # -- DQN hyperparameters --
    total_timesteps: int = args.timesteps
    lr: float = args.lr
    buffer_size: int = getattr(args, "buffer_size", 50_000)
    gamma: float = getattr(args, "gamma", 0.99)
    tau: float = getattr(args, "tau", 1.0)  # hard update by default
    target_network_frequency: int = getattr(args, "target_network_frequency", 500)
    batch_size: int = args.batch_size
    start_e: float = getattr(args, "start_e", 1.0)
    end_e: float = getattr(args, "end_e", 0.05)
    exploration_fraction: float = getattr(args, "exploration_fraction", 0.5)
    learning_starts: int = getattr(args, "learning_starts", 1000)
    train_frequency: int = getattr(args, "train_frequency", 4)

    # -- Logging --
    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="DQN",
        library="CleanRL",
        total_timesteps=total_timesteps,
        learning_rate=lr,
        num_envs=args.num_envs,
        batch_size=batch_size,
        buffer_size=buffer_size,
        gamma=gamma,
        tau=tau,
        target_network_frequency=target_network_frequency,
        start_epsilon=start_e,
        end_epsilon=end_e,
        exploration_fraction=exploration_fraction,
    )
    writer = SummaryWriter(log_dir / "tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    # -- Networks --
    q_network = QNetwork(obs_dim, act_dim).to(device)
    target_network = QNetwork(obs_dim, act_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)

    # -- Replay Buffer --
    rb = ReplayBuffer(buffer_size, obs_dim, device)

    # -- Init --
    global_step = 0
    start_time = time.time()
    next_obs_np, _ = env.reset()
    next_obs_flat = next_obs_np.reshape(num_envs_total, -1)

    # Episode stat accumulators
    ep_rewards = np.zeros(num_envs_total, dtype=np.float64)
    ep_lengths = np.zeros(num_envs_total, dtype=np.int64)
    ep_foods_collected = np.zeros(num_envs_total, dtype=np.int64)
    ep_coop_collections = np.zeros(num_envs_total, dtype=np.int64)
    ep_solo_collections = np.zeros(num_envs_total, dtype=np.int64)
    ep_failed_loads = np.zeros(num_envs_total, dtype=np.int64)
    ep_collisions = np.zeros(num_envs_total, dtype=np.int64)
    ep_food_remaining = np.zeros(num_envs_total, dtype=np.int64)
    ep_action_counts: list[dict[str, int]] = [{} for _ in range(num_envs_total)]

    print(f"CleanRL DQN | {total_timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  num_envs_total={num_envs_total}")
    print(f"  buffer_size={buffer_size}  batch_size={batch_size}")
    print(f"  device={device}")
    print()

    for global_step in range(1, total_timesteps + 1):
        # -- Epsilon-greedy action selection --
        epsilon = linear_schedule(
            start_e, end_e, int(exploration_fraction * total_timesteps), global_step
        )

        obs_tensor = torch.tensor(next_obs_flat, dtype=torch.float32, device=device)
        if random.random() < epsilon:
            actions = np.array([env.action_space.sample() for _ in range(num_envs_total)])
        else:
            with torch.no_grad():
                q_values = q_network(obs_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()

        # -- Step environment --
        new_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(actions)
        done_np = np.logical_or(terminated_np, truncated_np)
        new_obs_flat = new_obs_np.reshape(num_envs_total, -1)

        # Store transitions in replay buffer (one per sub-env)
        for i in range(num_envs_total):
            rb.add(
                next_obs_flat[i],
                new_obs_flat[i],
                int(actions[i]),
                float(reward_np[i]),
                bool(done_np[i]),
            )

        # Accumulate episode stats
        ep_rewards += np.array(reward_np, dtype=np.float64)
        ep_lengths += 1

        for i in range(num_envs_total):
            info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
            if isinstance(infos, dict):
                info_i = infos
            if isinstance(info_i, dict):
                ep_foods_collected[i] += info_i.get("foods_collected", 0)
                ep_coop_collections[i] += info_i.get("cooperative_collections", 0)
                ep_solo_collections[i] += info_i.get("solo_collections", 0)
                ep_failed_loads[i] += info_i.get("failed_loads", 0)
                ep_collisions[i] += info_i.get("collisions", 0)
                ep_food_remaining[i] = info_i.get("food_remaining", 0)
                for act_name, cnt in info_i.get("action_counts", {}).items():
                    ep_action_counts[i][act_name] = ep_action_counts[i].get(act_name, 0) + cnt

        # Check for finished episodes
        for i in range(num_envs_total):
            if done_np[i]:
                ep_stats: dict[str, Any] = {
                    "reward_total": float(ep_rewards[i]),
                    "length": int(ep_lengths[i]),
                    "foods_collected": int(ep_foods_collected[i]),
                    "cooperative_collections": int(ep_coop_collections[i]),
                    "solo_collections": int(ep_solo_collections[i]),
                    "failed_loads": int(ep_failed_loads[i]),
                    "food_remaining_end": int(ep_food_remaining[i]),
                    "collisions": int(ep_collisions[i]),
                    "action_counts": ep_action_counts[i],
                    "agent_rewards": {},
                }
                tracker.on_episode_end(global_step, ep_stats)
                writer.add_scalar("charts/episodic_return", ep_stats["reward_total"], global_step)
                writer.add_scalar("charts/episodic_length", ep_stats["length"], global_step)
                writer.add_scalar("charts/foods_collected", ep_stats["foods_collected"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
                ep_foods_collected[i] = 0
                ep_coop_collections[i] = 0
                ep_solo_collections[i] = 0
                ep_failed_loads[i] = 0
                ep_collisions[i] = 0
                ep_food_remaining[i] = 0
                ep_action_counts[i] = {}

        next_obs_flat = new_obs_flat

        # -- Training --
        if global_step > learning_starts and global_step % train_frequency == 0:
            s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(batch_size)

            with torch.no_grad():
                target_max, _ = target_network(s_next_obs).max(dim=1)
                td_target = s_rewards + gamma * target_max * (1 - s_dones)

            old_val = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze()
            loss = F.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -- Target network update --
        if global_step > learning_starts and global_step % target_network_frequency == 0:
            if tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
            else:
                for target_param, param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # -- Periodic logging --
        if global_step % max(1, total_timesteps // 20) == 0:
            sps = int(global_step / (time.time() - start_time))
            current_loss = loss.item() if global_step > learning_starts else 0.0
            print(
                f"  Steps: {global_step:>8,}/{total_timesteps:,} | "
                f"SPS: {sps:>5} | "
                f"Loss: {current_loss:>7.4f} | "
                f"Epsilon: {epsilon:.4f}"
            )
            writer.add_scalar("charts/SPS", sps, global_step)
            if global_step > learning_starts:
                writer.add_scalar("losses/td_loss", loss.item(), global_step)
                with torch.no_grad():
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )

    # -- Save model --
    model_path = log_dir / "model.pt"
    torch.save(q_network.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Visualize with:  uv run python -m scripts.compare_algorithms logs/{run_name}")

    tracker.close()
    writer.close()
    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser: argparse.ArgumentParser = get_standard_parser(
        description="Train DQN (CleanRL) on Sustainable Foraging"
    )
    # Override generic benchmark batch size to a standard DQN batch size
    parser.set_defaults(batch_size=128)
    # DQN-specific extra args
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size (default: 50000)")
    parser.add_argument("--tau", type=float, default=1.0, help="Target network update rate (1.0=hard update)")
    parser.add_argument(
        "--target-network-frequency", type=int, default=500,
        help="Steps between target network updates (default: 500)",
    )
    parser.add_argument("--start-e", type=float, default=1.0, help="Start epsilon (default: 1.0)")
    parser.add_argument("--end-e", type=float, default=0.05, help="End epsilon (default: 0.05)")
    parser.add_argument(
        "--exploration-fraction", type=float, default=0.5,
        help="Fraction of total steps for epsilon decay (default: 0.5)",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=1000,
        help="Steps before training begins (default: 1000)",
    )
    parser.add_argument(
        "--train-frequency", type=int, default=4,
        help="Steps between training updates (default: 4)",
    )

    train(parser.parse_args())
