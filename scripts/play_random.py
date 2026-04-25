import argparse
import logging
import time

import gymnasium as gym
import sustainable_foraging  # noqa
import numpy as np
from sustainable_foraging.foraging import AECForagingEnv

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.unwrapped.n_agents)

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        actions = env.action_space.sample()

        obss, rewards, done, _, _ = env.step(actions)
        returns += rewards

        if render:
            env.render()
            time.sleep(0.5)

    print("Returns: ", returns)


def _parse_env_id(env_id: str):
    """Parse simple env ids like Foraging-8x8-2p-2f-v3 -> (size, players, foods).
    Falls back to (8,2,2) on failure.
    """
    import re

    m = re.search(r"(\d+)x\d+-(\d+)p-(\d+)f", env_id)
    if not m:
        return 8, 2, 2
    size = int(m.group(1))
    players = int(m.group(2))
    foods = int(m.group(3))
    return size, players, foods


def _game_loop_aec(env, render):
    """Run an automatic random-agent loop for a PettingZoo AEC env."""
    env.reset()
    if render:
        env.render()
        time.sleep(0.5)

    # Loop until all agents are done (AEC env empties env.agents)
    while getattr(env, "agents", None):
        agent = env.agent_selection
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            action = None
        else:
            # sample from the agent's action space
            try:
                action = env.action_spaces[agent].sample()
            except Exception:
                # fallback to a discrete sample
                action = 0

        env.step(action)

        # Only render after all agents have acted (when the cycle wraps around)
        # In this specific env, the cycle order is deterministic (player_0, player_1, ...)
        # We render when we've just stepped the last agent.
        if render and agent == env.possible_agents[-1]:
            env.render()
            time.sleep(0.5)


def main(episodes=1, render=False, use_aec=False, env_id="Foraging-8x8-2p-2f-v3"):
    if use_aec:
        # instantiate and run the AEC env directly
        size, players, foods = _parse_env_id(env_id)
        env = AECForagingEnv(
            players=players,
            min_player_level=1,
            max_player_level=2,
            min_food_level=1,
            max_food_level=None,
            field_size=(size, size),
            max_num_food=foods,
            sight=size,
            max_episode_steps=50,
            force_coop=False,
        )

        for ep in range(1, episodes + 1):
            # reset returns for this episode
            ep_returns = {a: 0.0 for a in env.possible_agents}
            env.reset()

            # run until all agents are done
            while getattr(env, "agents", None):
                agent = env.agent_selection
                obs, _, terminated, truncated, _ = env.last()

                if terminated or truncated:
                    action = None
                else:
                    try:
                        action = env.action_spaces[agent].sample()
                    except Exception:
                        action = 0

                env.step(action)

                # accumulate per-step rewards (env.rewards holds latest step rewards)
                for a, r in getattr(env, "rewards", {}).items():
                    if r is None:
                        continue
                    ep_returns[a] = ep_returns.get(a, 0.0) + float(r)

                if render and agent == env.possible_agents[-1]:
                    env.render()
                    time.sleep(0.5)

            # episode finished
            total = sum(ep_returns.values())
            print(
                f"Episode {ep}/{episodes} finished — total return: {total:.3f} — per-agent: {ep_returns}"
            )

        env.close()
    else:
        # gym-registered envs (default)
        env = gym.make(env_id)
        for _ in range(episodes):
            _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random-agent rollouts in Gym or AEC envs.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run")
    parser.add_argument("--aec", action="store_true", help="Use PettingZoo AEC env")
    parser.add_argument(
        "--env",
        type=str,
        default="Foraging-8x8-2p-2f-v3",
        help="Env id to use (registered id or Foraging-... for AEC parsing)",
    )

    args = parser.parse_args()
    main(args.episodes, args.render, use_aec=args.aec, env_id=args.env)
