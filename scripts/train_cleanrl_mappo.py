#!/usr/bin/env python3
"""Train MAPPO (Multi-Agent PPO) on Sustainable Foraging using CleanRL-style code.

Single-file, dependency-light implementation following CleanRL conventions:
  - Pure PyTorch + gymnasium/PettingZoo
  - Parameter sharing: one policy network for all homogeneous agents
  - GAE for advantage estimation
  - Consistent CSV / config logging via ``_bench_utils``

Logs training metrics to:
  - CSV file                 (./logs/<run_name>/metrics.csv)
  - Experiment config        (./logs/<run_name>/config.json)
  - TensorBoard events       (./logs/<run_name>/tb/)
  - Saved model              (./logs/<run_name>/model.pt)
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import get_preset, get_training_defaults
from pettingzoo.utils import aec_to_parallel
from scripts._bench_utils import MetricsTracker, get_standard_parser, save_experiment_config

# ---------------------------------------------------------------------------
# Hyperparameter defaults (CleanRL-style)
# ---------------------------------------------------------------------------
TRAINING_DEFAULTS = get_training_defaults()


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal initialisation, standard for PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-Critic MLP for discrete action spaces (parameter-shared MAPPO)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(preset: str = "fair", num_envs: int = 1):
    """Create vectorized env using the same wrapping chain as SB3."""
    env_config = get_preset(preset)
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
def train(args: argparse.Namespace) -> None:
    run_name: str = args.name or f"cleanrl_mappo_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # -- Seeding --
    seed = getattr(args, "seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu")

    # -- Environment --
    env, env_config = make_env(args.preset, num_envs=args.num_envs)
    single_obs_space = env.observation_space
    single_act_space = env.action_space
    obs_dim = int(np.prod(single_obs_space.shape))
    act_dim = int(single_act_space.n)
    num_envs_total = env.num_envs  # num_envs * agents_per_env

    # -- PPO hyperparameters --
    total_timesteps: int = args.timesteps
    num_steps: int = getattr(args, "num_steps", 128)  # rollout length
    gamma: float = getattr(args, "gamma", 0.99)
    gae_lambda: float = getattr(args, "gae_lambda", 0.95)
    num_minibatches: int = getattr(args, "num_minibatches", 4)
    update_epochs: int = getattr(args, "update_epochs", 4)
    clip_coef: float = getattr(args, "clip_coef", 0.2)
    ent_coef: float = getattr(args, "ent_coef", 0.01)
    vf_coef: float = getattr(args, "vf_coef", 0.5)
    max_grad_norm: float = getattr(args, "max_grad_norm", 0.5)
    lr: float = args.lr
    anneal_lr: bool = getattr(args, "anneal_lr", True)

    batch_size = num_envs_total * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    # -- Logging --
    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="MAPPO",
        library="CleanRL",
        total_timesteps=total_timesteps,
        learning_rate=lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        num_steps=num_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )
    writer = SummaryWriter(log_dir / "tb")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(
            [f"|{k}|{v}|" for k, v in vars(args).items()]
        ),
    )

    # -- Agent --
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # -- Rollout storage --
    obs_buf = torch.zeros((num_steps, num_envs_total, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs_total), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs_total), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs_total), device=device)
    dones_buf = torch.zeros((num_steps, num_envs_total), device=device)
    values_buf = torch.zeros((num_steps, num_envs_total), device=device)

    # -- Init --
    global_step = 0
    start_time = time.time()
    next_obs_np, _ = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_envs_total, -1)
    next_done = torch.zeros(num_envs_total, device=device)

    # Track episode stats — accumulate per-step info across the episode
    ep_rewards = np.zeros(num_envs_total, dtype=np.float64)
    ep_lengths = np.zeros(num_envs_total, dtype=np.int64)
    ep_foods_collected = np.zeros(num_envs_total, dtype=np.int64)
    ep_coop_collections = np.zeros(num_envs_total, dtype=np.int64)
    ep_solo_collections = np.zeros(num_envs_total, dtype=np.int64)
    ep_failed_loads = np.zeros(num_envs_total, dtype=np.int64)
    ep_collisions = np.zeros(num_envs_total, dtype=np.int64)
    ep_food_remaining = np.zeros(num_envs_total, dtype=np.int64)
    ep_action_counts: list[dict[str, int]] = [{} for _ in range(num_envs_total)]

    print(f"CleanRL MAPPO | {total_timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  num_envs_total={num_envs_total}")
    print(f"  batch_size={batch_size}  minibatch_size={minibatch_size}  num_updates={num_updates}")
    print(f"  device={device}")
    print()

    for update in range(1, num_updates + 1):
        # -- Anneal LR --
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * lr

        # -- Rollout phase --
        for step in range(num_steps):
            global_step += num_envs_total
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step environment
            next_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated_np, truncated_np)
            rewards_buf[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(
                num_envs_total, -1
            )
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            # Accumulate episode stats every step (not just on done)
            ep_rewards += np.array(reward_np, dtype=np.float64)
            ep_lengths += 1

            # Extract per-step info and accumulate across the episode
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
                    ep_food_remaining[i] = info_i.get("food_remaining", 0)  # snapshot, not sum
                    for act_name, cnt in info_i.get("action_counts", {}).items():
                        ep_action_counts[i][act_name] = ep_action_counts[i].get(act_name, 0) + cnt

            # Check for finished episodes and flush accumulated stats
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

                    # Reset accumulators for this slot
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    ep_foods_collected[i] = 0
                    ep_coop_collections[i] = 0
                    ep_solo_collections[i] = 0
                    ep_failed_loads[i] = 0
                    ep_collisions[i] = 0
                    ep_food_remaining[i] = 0
                    ep_action_counts[i] = {}

        # -- GAE --
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # -- Flatten batch --
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # -- PPO update --
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # -- Logging --
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        if update % max(1, num_updates // 20) == 0 or update == num_updates:
            print(
                f"  Update {update:>4}/{num_updates} | "
                f"Steps: {global_step:>8,}/{total_timesteps:,} | "
                f"SPS: {sps:>5} | "
                f"Loss: {loss.item():>7.4f} | "
                f"Entropy: {entropy_loss.item():.4f} | "
                f"Explained Var: {explained_var:.4f}"
            )

    # -- Save model --
    model_path = log_dir / "model.pt"
    torch.save(agent.state_dict(), model_path)
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
        description="Train MAPPO (CleanRL) on Sustainable Foraging"
    )
    # CleanRL-specific extra args
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout length per env (default: 128)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (default: 0.95)")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches (default: 4)")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs (default: 4)")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient (default: 0.2)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm (default: 0.5)")
    parser.add_argument("--anneal-lr", action="store_true", default=True, help="Anneal learning rate")
    parser.add_argument("--no-anneal-lr", action="store_false", dest="anneal_lr", help="Disable LR annealing")

    train(parser.parse_args())
