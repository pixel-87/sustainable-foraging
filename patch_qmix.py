
def _run_qmix(args):
    run_name = args.name or f"cleanrl_qmix_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env_config = {"grid_observation": False}
    env, full_config = make_env(args.preset, num_envs=args.num_envs, config_overrides=env_config, vectorize_for_cleanrl_sb3=True)
    
    raw_env = AECForagingEnv(**full_config)
    n_agents = len(raw_env.possible_agents)
    raw_env.close()

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    state_dim = n_agents * obs_dim
    num_envs_total = env.num_envs

    tracker = MetricsTracker(log_dir / "metrics.csv")
    save_experiment_config(
        log_dir, run_name, args.preset, algorithm="QMIX", library="CleanRL",
        total_timesteps=args.timesteps, learning_rate=args.lr, num_envs=args.num_envs,
        batch_size=args.batch_size, buffer_size=args.buffer_size, gamma=args.gamma,
        tau=args.tau, target_network_frequency=args.target_network_frequency,
        start_epsilon=args.start_e, end_epsilon=args.end_e, exploration_fraction=args.exploration_fraction,
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
        epsilon = linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.timesteps), global_step)

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
        env_dones = done_np.reshape(args.num_envs, n_agents).all(axis=1) # Global done when all agents are done

        for i in range(args.num_envs):
            rb.add(next_obs_flat[i], new_obs_flat[i], next_state_flat[i], new_state_flat[i], actions[i], float(env_rewards[i]), bool(env_dones[i]))
            
            # Agent i * n_agents is the first agent of the parallel env i, which should hold episode_metrics from wrapper
            idx = i * n_agents
            info_i = infos[idx] if isinstance(infos, (list, tuple)) else infos
            if isinstance(infos, dict): info_i = infos
            if "episode_metrics" in info_i:
                metrics = info_i["episode_metrics"]
                tracker.on_episode_end(global_step, metrics)
                writer.add_scalar("charts/episodic_return", metrics["reward_total"], global_step)
                writer.add_scalar("charts/episodic_length", metrics["length"], global_step)

        next_obs_flat = new_obs_flat
        next_state_flat = new_state_flat

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_next_obs, s_states, s_next_states, s_actions, s_rewards, s_dones = rb.sample(args.batch_size)
            
            # Reshape for individual Q networks
            s_obs_batch = s_obs.view(-1, obs_dim)
            s_next_obs_batch = s_next_obs.view(-1, obs_dim)

            # Get current Q values
            mac_out = q_network(s_obs_batch).view(args.batch_size, n_agents, act_dim)
            chosen_action_qvals = torch.gather(mac_out, dim=2, index=s_actions.unsqueeze(2)).squeeze(2)

            with torch.no_grad():
                target_mac_out = target_network(s_next_obs_batch).view(args.batch_size, n_agents, act_dim)
                target_max_qvals = target_mac_out.max(dim=2)[0]
                
                # Mixing target
                target_tot = target_mixer(target_max_qvals, s_next_states)
                td_target = s_rewards.unsqueeze(1) + args.gamma * target_tot * (1 - s_dones.unsqueeze(1))

            # Mixing current
            chosen_action_qvals_tot = mixer(chosen_action_qvals, s_states)
            loss = F.mse_loss(chosen_action_qvals_tot, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            if args.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
            else:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                for target_param, param in zip(target_mixer.parameters(), mixer.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

        if global_step % max(1, args.timesteps // 20) == 0:
            print(f"  Steps: {global_step:>8,}/{args.timesteps:,} | Epsilon: {epsilon:.4f}")

    model_path = log_dir / "model.pt"
    torch.save({"q_net": q_network.state_dict(), "mixer": mixer.state_dict()}, model_path)
    tracker.close()
    writer.close()
    env.close()
