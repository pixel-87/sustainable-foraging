from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sustainable_foraging.foraging import AECForagingEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from scripts._bench_utils import MetricsTracker, save_experiment_config
from scripts.core.env_utils import make_env


# ---------------------------------------------------------------------------
# MAPPO Network
# ---------------------------------------------------------------------------
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOAgent(nn.Module):
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
        probs = Categorical(logits=logits, validate_args=False)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# ---------------------------------------------------------------------------
# DQN Network & Buffer
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
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


class ReplayBuffer:
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

    def add(self, obs, next_obs, action, reward, done):
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run_cleanrl(args: argparse.Namespace, algorithm: str) -> None:
    if algorithm == "mappo":
        _run_mappo(args)
    elif algorithm == "dqn":
        _run_dqn(args)
    elif algorithm == "vdn":
        _run_vdn(args)
    elif algorithm == "qmix":
        _run_qmix(args)
    else:
        raise ValueError(f"CleanRL doesn't support {algorithm}")


def _run_mappo(args: argparse.Namespace) -> None:
    run_name = args.name or f"cleanrl_mappo_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env, _ = make_env(
        args.preset, num_envs=args.num_envs, vectorize_for_cleanrl_sb3=True, num_cpus=args.num_cpus
    )
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    num_envs_total = env.num_envs

    batch_size = num_envs_total * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.timesteps // batch_size

    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="MAPPO",
        library="CleanRL",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )
    writer = SummaryWriter(log_dir / "tb")

    agent = MAPPOAgent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, num_envs_total, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, num_envs_total), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((args.num_steps, num_envs_total), device=device)
    rewards_buf = torch.zeros((args.num_steps, num_envs_total), device=device)
    dones_buf = torch.zeros((args.num_steps, num_envs_total), device=device)
    values_buf = torch.zeros((args.num_steps, num_envs_total), device=device)

    global_step = 0
    time.time()
    next_obs_np, _ = env.reset()
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(
        num_envs_total, -1
    )
    next_done = torch.zeros(num_envs_total, device=device)

    print(f"CleanRL MAPPO | {args.timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  num_envs_total={num_envs_total}")
    print(f"  batch_size={batch_size}  minibatch_size={minibatch_size}  num_updates={num_updates}")
    print(f"  device={device}\n")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.lr

        for step in range(args.num_steps):
            global_step += num_envs_total
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated_np, truncated_np)
            rewards_buf[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device).reshape(
                num_envs_total, -1
            )
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

            for i in range(num_envs_total):
                info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
                if isinstance(infos, dict):
                    info_i = infos
                if "episode_metrics" in info_i:
                    metrics = info_i["episode_metrics"]
                    tracker.on_episode_end(global_step, metrics)
                    writer.add_scalar(
                        "charts/episodic_return", metrics["reward_total"], global_step
                    )
                    writer.add_scalar("charts/episodic_length", metrics["length"], global_step)
                    writer.add_scalar(
                        "charts/foods_collected", metrics["foods_collected"], global_step
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _epoch in range(args.update_epochs):
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
                    ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        if update % max(1, num_updates // 20) == 0 or update == num_updates:
            print(
                f"  Update {update:>4}/{num_updates} | Steps: {global_step:>8,}/{args.timesteps:,}"
            )

    model_path = log_dir / "model.pt"
    torch.save(agent.state_dict(), model_path)
    tracker.close()
    writer.close()
    env.close()


def _run_dqn(args: argparse.Namespace) -> None:
    run_name = args.name or f"cleanrl_dqn_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env, _ = make_env(
        args.preset,
        num_envs=args.num_envs,
        config_overrides={"grid_observation": False},
        vectorize_for_cleanrl_sb3=True,
        num_cpus=args.num_cpus,
    )
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    num_envs_total = env.num_envs

    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="DQN",
        library="CleanRL",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        target_network_frequency=args.target_network_frequency,
        start_epsilon=args.start_e,
        end_epsilon=args.end_e,
        exploration_fraction=args.exploration_fraction,
    )
    writer = SummaryWriter(log_dir / "tb")

    q_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, obs_dim, device)

    global_step = 0
    next_obs_np, _ = env.reset()
    next_obs_flat = next_obs_np.reshape(num_envs_total, -1)

    print(f"CleanRL DQN | {args.timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  num_envs_total={num_envs_total}")
    print(f"  buffer_size={args.buffer_size}  batch_size={args.batch_size}")
    print(f"  device={device}\n")

    for global_step in range(1, args.timesteps + 1):
        epsilon = linear_schedule(
            args.start_e, args.end_e, int(args.exploration_fraction * args.timesteps), global_step
        )

        obs_tensor = torch.tensor(next_obs_flat, dtype=torch.float32, device=device)
        if random.random() < epsilon:
            actions = np.array([env.action_space.sample() for _ in range(num_envs_total)])
        else:
            with torch.no_grad():
                actions = q_network(obs_tensor).argmax(dim=1).cpu().numpy()

        new_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(actions)
        done_np = np.logical_or(terminated_np, truncated_np)
        new_obs_flat = new_obs_np.reshape(num_envs_total, -1)

        for i in range(num_envs_total):
            rb.add(
                next_obs_flat[i],
                new_obs_flat[i],
                int(actions[i]),
                float(reward_np[i]),
                bool(done_np[i]),
            )

            info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
            if isinstance(infos, dict):
                info_i = infos
            if "episode_metrics" in info_i:
                metrics = info_i["episode_metrics"]
                tracker.on_episode_end(global_step, metrics)
                writer.add_scalar("charts/episodic_return", metrics["reward_total"], global_step)
                writer.add_scalar("charts/episodic_length", metrics["length"], global_step)

        next_obs_flat = new_obs_flat

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(s_next_obs).max(dim=1)
                td_target = s_rewards + args.gamma * target_max * (1 - s_dones)
            old_val = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze()
            loss = F.mse_loss(old_val, td_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(q_network.parameters()) + list(mixer.parameters())
                if "mixer" in locals()
                else q_network.parameters(),
                getattr(args, "max_grad_norm", 10.0),
            )
            optimizer.step()

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            if args.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
            else:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

        if global_step % max(1, args.timesteps // 20) == 0:
            current_loss = loss.item() if "loss" in locals() else 0.0
            print(
                f"  Steps: {global_step:>8,}/{args.timesteps:,} | Loss: {current_loss:>7.4f} | Epsilon: {epsilon:.4f}"
            )

    model_path = log_dir / "model.pt"
    torch.save(q_network.state_dict(), model_path)
    tracker.close()
    writer.close()
    env.close()


# ---------------------------------------------------------------------------
# QMIX Network & Buffer
# ---------------------------------------------------------------------------
class QMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, self.embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.embed_dim, self.embed_dim * self.n_agents), std=0.01),
        )
        self.hyper_w_final = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, self.embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.embed_dim, self.embed_dim), std=0.01),
        )

        self.hyper_b_1 = layer_init(nn.Linear(self.state_dim, self.embed_dim))
        self.V = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, self.embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.embed_dim, 1)),
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1)
        return q_tot


class QMixReplayBuffer:
    def __init__(
        self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, device: torch.device
    ):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, next_obs, state, next_state, action, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.state[self.pos] = state
        self.next_state[self.pos] = next_state
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
            torch.as_tensor(self.state[idxs], device=self.device),
            torch.as_tensor(self.next_state[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device).long(),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.dones[idxs], device=self.device),
        )


def _run_qmix(args: argparse.Namespace) -> None:
    run_name = args.name or f"cleanrl_qmix_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env_config = {"grid_observation": False}
    env, full_config = make_env(
        args.preset,
        num_envs=args.num_envs,
        config_overrides=env_config,
        vectorize_for_cleanrl_sb3=True,
        num_cpus=args.num_cpus,
    )

    raw_env = AECForagingEnv(**full_config)
    n_agents = len(raw_env.possible_agents)
    raw_env.close()

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    state_dim = n_agents * obs_dim
    num_envs_total = env.num_envs

    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="QMIX",
        library="CleanRL",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        target_network_frequency=args.target_network_frequency,
        start_epsilon=args.start_e,
        end_epsilon=args.end_e,
        exploration_fraction=args.exploration_fraction,
    )
    writer = SummaryWriter(log_dir / "tb")

    # QMIX networks
    q_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    mixer = QMixer(n_agents, state_dim).to(device)
    target_mixer = QMixer(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    optimizer = optim.Adam(list(q_network.parameters()) + list(mixer.parameters()), lr=args.lr)
    rb = QMixReplayBuffer(args.buffer_size, n_agents, obs_dim, state_dim, device)

    global_step = 0
    next_obs_np, _ = env.reset()
    next_obs_flat = next_obs_np.reshape(args.num_envs, n_agents, obs_dim)
    next_state_flat = next_obs_flat.reshape(args.num_envs, state_dim)

    print(f"CleanRL QMIX | {args.timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  state_dim={state_dim}  act_dim={act_dim}  n_agents={n_agents}")
    print(f"  buffer_size={args.buffer_size}  batch_size={args.batch_size}")
    print(f"  device={device}\n")

    for global_step in range(1, args.timesteps + 1):
        epsilon = linear_schedule(
            args.start_e, args.end_e, int(args.exploration_fraction * args.timesteps), global_step
        )

        obs_tensor = torch.tensor(next_obs_flat, dtype=torch.float32, device=device)
        actions = np.zeros((args.num_envs, n_agents), dtype=np.int64)

        if random.random() < epsilon:
            actions_flat = np.array([env.action_space.sample() for _ in range(num_envs_total)])
            actions = actions_flat.reshape(args.num_envs, n_agents)
        else:
            with torch.no_grad():
                # Q-network input: [num_envs * n_agents, obs_dim]
                q_vals = q_network(obs_tensor.view(-1, obs_dim))
                actions_flat = q_vals.argmax(dim=1).cpu().numpy()
                actions = actions_flat.reshape(args.num_envs, n_agents)

        new_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(actions.flatten())
        done_np = np.logical_or(terminated_np, truncated_np)

        new_obs_flat = new_obs_np.reshape(args.num_envs, n_agents, obs_dim)
        new_state_flat = new_obs_flat.reshape(args.num_envs, state_dim)

        # In PZ AEC->Parallel, rewards are given per-agent. For QMIX, we sum them per environment to get global reward.
        env_rewards = reward_np.reshape(args.num_envs, n_agents).sum(axis=1)
        env_dones = done_np.reshape(args.num_envs, n_agents).any(
            axis=1
        )  # Global done when any agent is done

        for i in range(args.num_envs):
            rb.add(
                next_obs_flat[i],
                new_obs_flat[i],
                next_state_flat[i],
                new_state_flat[i],
                actions[i],
                float(env_rewards[i]),
                bool(env_dones[i]),
            )

            # Agent i * n_agents is the first agent of the parallel env i, which should hold episode_metrics from wrapper
            idx = i * n_agents
            info_i = infos[idx] if isinstance(infos, (list, tuple)) else infos
            if isinstance(infos, dict):
                info_i = infos
            if "episode_metrics" in info_i:
                metrics = info_i["episode_metrics"]
                tracker.on_episode_end(global_step, metrics)
                writer.add_scalar("charts/episodic_return", metrics["reward_total"], global_step)
                writer.add_scalar("charts/episodic_length", metrics["length"], global_step)

        next_obs_flat = new_obs_flat
        next_state_flat = new_state_flat

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_next_obs, s_states, s_next_states, s_actions, s_rewards, s_dones = rb.sample(
                args.batch_size
            )

            # Reshape for individual Q networks
            s_obs_batch = s_obs.view(-1, obs_dim)
            s_next_obs_batch = s_next_obs.view(-1, obs_dim)

            # Get current Q values
            mac_out = q_network(s_obs_batch).view(args.batch_size, n_agents, act_dim)
            chosen_action_qvals = torch.gather(
                mac_out, dim=2, index=s_actions.unsqueeze(2)
            ).squeeze(2)

            with torch.no_grad():
                mac_out_next = q_network(s_next_obs_batch).view(args.batch_size, n_agents, act_dim)
                next_actions = mac_out_next.argmax(dim=2, keepdim=True)
                target_mac_out = target_network(s_next_obs_batch).view(
                    args.batch_size, n_agents, act_dim
                )
                target_max_qvals = torch.gather(target_mac_out, dim=2, index=next_actions).squeeze(
                    2
                )

                # Mixing target
                target_tot = target_mixer(target_max_qvals, s_next_states)
                td_target = s_rewards.unsqueeze(1) + args.gamma * target_tot * (
                    1 - s_dones.unsqueeze(1)
                )

            # Mixing current
            chosen_action_qvals_tot = mixer(chosen_action_qvals, s_states)
            loss = F.mse_loss(chosen_action_qvals_tot, td_target.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(q_network.parameters()) + list(mixer.parameters())
                if "mixer" in locals()
                else q_network.parameters(),
                getattr(args, "max_grad_norm", 10.0),
            )
            optimizer.step()

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            if args.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
            else:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )
                for target_param, param in zip(target_mixer.parameters(), mixer.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

        if global_step % max(1, args.timesteps // 20) == 0:
            current_loss = loss.item() if "loss" in locals() else 0.0
            print(
                f"  Steps: {global_step:>8,}/{args.timesteps:,} | Loss: {current_loss:>7.4f} | Epsilon: {epsilon:.4f}"
            )

    model_path = log_dir / "model.pt"
    torch.save({"q_net": q_network.state_dict(), "mixer": mixer.state_dict()}, model_path)
    tracker.close()
    writer.close()
    env.close()


class VDNMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, agent_qs, states):
        # VDN simply sums the Q-values of all agents
        # agent_qs shape: [batch_size, 1, n_agents] or [batch_size, n_agents]
        # We want to return [batch_size, 1] to match QMixer output shape
        if len(agent_qs.shape) == 3:
            return agent_qs.sum(dim=2)
        return agent_qs.sum(dim=1, keepdim=True)


def _run_vdn(args: argparse.Namespace) -> None:
    run_name = args.name or f"cleanrl_vdn_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env_config = {"grid_observation": False}
    env, full_config = make_env(
        args.preset,
        num_envs=args.num_envs,
        config_overrides=env_config,
        vectorize_for_cleanrl_sb3=True,
        num_cpus=args.num_cpus,
    )

    raw_env = AECForagingEnv(**full_config)
    n_agents = len(raw_env.possible_agents)
    raw_env.close()

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    state_dim = n_agents * obs_dim
    num_envs_total = env.num_envs

    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir,
        run_name,
        args.preset,
        algorithm="VDN",
        library="CleanRL",
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        target_network_frequency=args.target_network_frequency,
        start_epsilon=args.start_e,
        end_epsilon=args.end_e,
        exploration_fraction=args.exploration_fraction,
    )
    writer = SummaryWriter(log_dir / "tb")

    # VDN networks
    q_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network = DQNNetwork(obs_dim, act_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    mixer = VDNMixer().to(device)
    target_mixer = VDNMixer().to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    optimizer = optim.Adam(list(q_network.parameters()) + list(mixer.parameters()), lr=args.lr)
    rb = QMixReplayBuffer(args.buffer_size, n_agents, obs_dim, state_dim, device)

    global_step = 0
    next_obs_np, _ = env.reset()
    next_obs_flat = next_obs_np.reshape(args.num_envs, n_agents, obs_dim)
    next_state_flat = next_obs_flat.reshape(args.num_envs, state_dim)

    print(f"CleanRL VDN | {args.timesteps:,} steps | preset={args.preset}")
    print(f"  obs_dim={obs_dim}  state_dim={state_dim}  act_dim={act_dim}  n_agents={n_agents}")
    print(f"  buffer_size={args.buffer_size}  batch_size={args.batch_size}")
    print(f"  device={device}\n")

    for global_step in range(1, args.timesteps + 1):
        epsilon = linear_schedule(
            args.start_e, args.end_e, int(args.exploration_fraction * args.timesteps), global_step
        )

        obs_tensor = torch.tensor(next_obs_flat, dtype=torch.float32, device=device)
        actions = np.zeros((args.num_envs, n_agents), dtype=np.int64)

        if random.random() < epsilon:
            actions_flat = np.array([env.action_space.sample() for _ in range(num_envs_total)])
            actions = actions_flat.reshape(args.num_envs, n_agents)
        else:
            with torch.no_grad():
                q_vals = q_network(obs_tensor.view(-1, obs_dim))
                actions_flat = q_vals.argmax(dim=1).cpu().numpy()
                actions = actions_flat.reshape(args.num_envs, n_agents)

        new_obs_np, reward_np, terminated_np, truncated_np, infos = env.step(actions.flatten())
        done_np = np.logical_or(terminated_np, truncated_np)

        new_obs_flat = new_obs_np.reshape(args.num_envs, n_agents, obs_dim)
        new_state_flat = new_obs_flat.reshape(args.num_envs, state_dim)

        env_rewards = reward_np.reshape(args.num_envs, n_agents).sum(axis=1)
        env_dones = done_np.reshape(args.num_envs, n_agents).any(axis=1)

        for i in range(args.num_envs):
            rb.add(
                next_obs_flat[i],
                new_obs_flat[i],
                next_state_flat[i],
                new_state_flat[i],
                actions[i],
                float(env_rewards[i]),
                bool(env_dones[i]),
            )

            idx = i * n_agents
            info_i = infos[idx] if isinstance(infos, (list, tuple)) else infos
            if isinstance(infos, dict):
                info_i = infos
            if "episode_metrics" in info_i:
                metrics = info_i["episode_metrics"]
                tracker.on_episode_end(global_step, metrics)
                writer.add_scalar("charts/episodic_return", metrics["reward_total"], global_step)
                writer.add_scalar("charts/episodic_length", metrics["length"], global_step)

        next_obs_flat = new_obs_flat
        next_state_flat = new_state_flat

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_next_obs, s_states, s_next_states, s_actions, s_rewards, s_dones = rb.sample(
                args.batch_size
            )

            s_obs_batch = s_obs.view(-1, obs_dim)
            s_next_obs_batch = s_next_obs.view(-1, obs_dim)

            mac_out = q_network(s_obs_batch).view(args.batch_size, n_agents, act_dim)
            chosen_action_qvals = torch.gather(
                mac_out, dim=2, index=s_actions.unsqueeze(2)
            ).squeeze(2)

            with torch.no_grad():
                mac_out_next = q_network(s_next_obs_batch).view(args.batch_size, n_agents, act_dim)
                next_actions = mac_out_next.argmax(dim=2, keepdim=True)
                target_mac_out = target_network(s_next_obs_batch).view(
                    args.batch_size, n_agents, act_dim
                )
                target_max_qvals = torch.gather(target_mac_out, dim=2, index=next_actions).squeeze(
                    2
                )

                target_tot = target_mixer(target_max_qvals, s_next_states)
                td_target = s_rewards.unsqueeze(1) + args.gamma * target_tot * (
                    1 - s_dones.unsqueeze(1)
                )

            chosen_action_qvals_tot = mixer(chosen_action_qvals, s_states)
            loss = F.mse_loss(chosen_action_qvals_tot, td_target.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(q_network.parameters()) + list(mixer.parameters())
                if "mixer" in locals()
                else q_network.parameters(),
                getattr(args, "max_grad_norm", 10.0),
            )
            optimizer.step()

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            if args.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
            else:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

        if global_step % max(1, args.timesteps // 20) == 0:
            current_loss = loss.item() if "loss" in locals() else 0.0
            print(
                f"  Steps: {global_step:>8,}/{args.timesteps:,} | Loss: {current_loss:>7.4f} | Epsilon: {epsilon:.4f}"
            )

    model_path = log_dir / "model.pt"
    torch.save({"q_net": q_network.state_dict(), "mixer": mixer.state_dict()}, model_path)
    tracker.close()
    writer.close()
    env.close()
