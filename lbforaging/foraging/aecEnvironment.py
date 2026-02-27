import logging
from collections import defaultdict, namedtuple
from enum import Enum
from itertools import product
from typing import ClassVar, Iterable

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.energy = None
        self.max_energy = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, max_energy, field_size):
        self.history = []
        self.position = position
        self.max_energy = max_energy
        self.energy = max_energy
        self.field_size = field_size
        self.score = 0

    @property
    def is_dead(self):
        return self.energy is not None and self.energy <= 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        if self.controller is None:
            raise ValueError("Controller not set for player")
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(AECEnv):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata: ClassVar[dict] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
        "is_parallelizable": True,
    }

    action_set: ClassVar[list] = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "energy", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        max_energy=50,
        food_energy_value=10,
        energy_depletion_rate=1,
        food_regeneration_rate=1.5,  # α: logistic replenishment rate (must be > 1)
        num_food_zones=2,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_energy=True,
        penalty=0.0,
        render_mode=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.viewer = None
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty

        self.max_energy = max_energy
        self.food_energy_value = food_energy_value
        self.energy_depletion_rate = energy_depletion_rate
        self.food_regeneration_rate = food_regeneration_rate
        self.num_food_zones = num_food_zones
        self.food_zones = []

        self.max_num_food = max_num_food  # K: carrying capacity
        self._food_spawned = 0.0
        self._food_level = 0.0  # continuous food level for logistic growth

        self.sight = sight
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = {}
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._observe_agent_energy = observe_agent_energy

        self.n_agents = len(self.players)

        # AEC Setup
        self.possible_agents = ["player_" + str(r) for r in range(self.n_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = AgentSelector(self.possible_agents)

        self.action_spaces = {agent: gym.spaces.Discrete(6) for agent in self.possible_agents}
        self.observation_spaces = {agent: self._get_observation_space() for agent in self.possible_agents}

        self._np_random, _ = seeding.np_random(None)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, energy)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_num_food = self.max_num_food

            if self._observe_agent_energy:
                min_obs = [-1, -1] * max_num_food + [-1, -1, 0] * len(self.players)
                max_obs = [field_x - 1, field_y - 1] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                    self.max_energy,
                ] * len(self.players)
            else:
                min_obs = [-1, -1] * max_num_food + [-1, -1] * len(self.players)
                max_obs = [field_x - 1, field_y - 1] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent energy
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            if self._observe_agent_energy:
                agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_energy
            else:
                agents_max = np.ones(grid_shape, dtype=np.float32)

            # foods layer: foods presence
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32)

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        low_obs = np.array(min_obs)
        high_obs = np.array(max_obs)
        assert low_obs.shape == high_obs.shape
        return gym.spaces.Box(
            low=low_obs, high=high_obs, shape=low_obs.shape, dtype=np.float32
        )

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.energy, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(
            players,
            field_size=None,
            max_num_food=None,
            sight=None,
            max_episode_steps=50,
        )

        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1
        return None  # No adjacent food found

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if player.position is not None
            and (
                (abs(player.position[0] - row) == 1 and player.position[1] == col)
                or (abs(player.position[1] - col) == 1 and player.position[0] == row)
            )
        ]

    def spawn_food(self, max_num_food):
        food_count = 0
        attempts = 0

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            
            # Pick a random food zone
            if self.food_zones:
                zone = self.food_zones[self._np_random.integers(0, len(self.food_zones))]
                # Spawn within a radius of 2 from the zone center
                row = np.clip(zone[0] + self._np_random.integers(-2, 3), 1, self.rows - 2)
                col = np.clip(zone[1] + self._np_random.integers(-2, 3), 1, self.cols - 2)
            else:
                row = self._np_random.integers(1, self.rows - 1)
                col = self._np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = 1
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_energy):
        for player in self.players:
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self._np_random.integers(0, self.rows)
                col = self._np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        max_energy,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if player.position is None:
            return action == Action.NONE
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error(f"Undefined action {action} from {player.name}")
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        if center is None or position is None:
            return (-1, -1)
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    energy=a.energy,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)) if player.position is not None else np.zeros((2 * self.sight + 1, 2 * self.sight + 1), np.int32),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_obs_array(self, observation):
        obs = np.zeros(self.observation_spaces[self.possible_agents[0]].shape, dtype=np.float32)
        # obs[: observation.field.size] = observation.field.flatten()
        # self player is always first
        seen_players = [p for p in observation.players if p.is_self] + [
            p for p in observation.players if not p.is_self
        ]

        for i in range(self.max_num_food):
            obs[2 * i] = -1
            obs[2 * i + 1] = -1

        for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
            obs[2 * i] = y
            obs[2 * i + 1] = x

        player_obs_len = 3 if self._observe_agent_energy else 2
        for i in range(len(self.players)):
            obs[self.max_num_food * 2 + player_obs_len * i] = -1
            obs[self.max_num_food * 2 + player_obs_len * i + 1] = -1
            if self._observe_agent_energy:
                obs[self.max_num_food * 2 + player_obs_len * i + 2] = 0

        for i, p in enumerate(seen_players):
            obs[self.max_num_food * 2 + player_obs_len * i] = p.position[0]
            obs[self.max_num_food * 2 + player_obs_len * i + 1] = p.position[1]
            if self._observe_agent_energy:
                obs[self.max_num_food * 2 + player_obs_len * i + 2] = p.energy

        return obs

    def _make_global_grid_arrays(self):
        """
        Create global arrays for grid observation space
        """
        grid_shape_x, grid_shape_y = self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        grid_shape = (grid_shape_x, grid_shape_y)

        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for player in self.players:
            if player.position is None:
                continue
            player_x, player_y = player.position
            if self._observe_agent_energy:
                agents_layer[player_x + self.sight, player_y + self.sight] = (
                    player.energy
                )
            else:
                agents_layer[player_x + self.sight, player_y + self.sight] = 1

        foods_layer = np.zeros(grid_shape, dtype=np.float32)
        foods_layer[self.sight : -self.sight, self.sight : -self.sight] = (
            self.field.copy()
        )

        access_layer = np.ones(grid_shape, dtype=np.float32)
        # out of bounds not accessible
        access_layer[: self.sight, :] = 0.0
        access_layer[-self.sight :, :] = 0.0
        access_layer[:, : self.sight] = 0.0
        access_layer[:, -self.sight :] = 0.0
        # agent locations are not accessible
        for player in self.players:
            if player.position is None:
                continue
            player_x, player_y = player.position
            access_layer[player_x + self.sight, player_y + self.sight] = 0.0
        # food locations are not accessible
        foods_x, foods_y = self.field.nonzero()
        for x, y in zip(foods_x, foods_y):
            access_layer[x + self.sight, y + self.sight] = 0.0

        return np.stack([agents_layer, foods_layer, access_layer])

    def _get_agent_grid_bounds(self, agent_x, agent_y):
        return (
            agent_x,
            agent_x + 2 * self.sight + 1,
            agent_y,
            agent_y + 2 * self.sight + 1,
        )

    def _make_gym_obs(self):
        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = self._make_global_grid_arrays()
            agents_bounds = [
                self._get_agent_grid_bounds(*player.position) if player.position is not None else None
                for player in self.players
            ]
            
            # Reorder to match players
            ordered_nobs = []
            for i, bounds in enumerate(agents_bounds):
                if bounds is not None:
                    start_x, end_x, start_y, end_y = bounds
                    ordered_nobs.append(layers[:, start_x:end_x, start_y:end_y])
                else:
                    ordered_nobs.append(np.zeros(self.observation_spaces[self.possible_agents[i]].shape, dtype=np.float32))
            nobs = tuple(ordered_nobs)
        else:
            nobs = tuple([self._make_obs_array(obs) for obs in observations])

        # check the space of obs
        for i, obs in enumerate(nobs):
            assert self.observation_spaces[self.possible_agents[i]].contains(
                obs
            ), f"obs space error: obs: {obs}, obs_space: {self.observation_spaces[self.possible_agents[i]]}"

        return nobs

    def _get_info(self):
        return {
            "foods_collected": self._step_foods_collected,
            "cooperative_collections": self._step_cooperative_collections,
            "solo_collections": self._step_solo_collections,
            "failed_loads": self._step_failed_loads,
            "collisions": self._step_collisions,
            "food_remaining": int(np.count_nonzero(self.field)),
            "food_total": self.max_num_food,
            "action_counts": dict(self._step_action_counts),
            "per_agent_rewards": [p.reward for p in self.players],
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
             self._np_random, seed = seeding.np_random(seed)

        self.field = np.zeros(self.field_size, np.int32)
        
        # Generate food zones
        self.food_zones = []
        for _ in range(self.num_food_zones):
            row = self._np_random.integers(2, self.rows - 2)
            col = self._np_random.integers(2, self.cols - 2)
            self.food_zones.append((row, col))

        self.spawn_players(self.max_energy)

        self.spawn_food(self.max_num_food)
        self._food_level = float(np.count_nonzero(self.field))  # init continuous level
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        # Step-level metric counters (reset each step in _process_actions)
        self._step_foods_collected = 0
        self._step_cooperative_collections = 0
        self._step_solo_collections = 0
        self._step_failed_loads = 0
        self._step_collisions = 0
        self._step_action_counts = {}

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._actions = {agent: None for agent in self.agents}

        # return self._make_gym_obs(), self._get_info()

    def observe(self, agent):
        """Return the observation for the specified agent."""
        obs = self._make_gym_obs()
        return obs[self.agent_name_mapping[agent]]

    def last(self, observe=True):
        """Return observation, cumulative reward, terminated, truncated, info for the current agent.

        This is the primary method for interacting with AEC environments.
        """
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action):
        """Execute one agent's action following the AEC buffered execution pattern."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self._actions[agent] = action

        if self._agent_selector.is_last():
            # Execute all buffered actions
            actions_list = [self._actions[a] for a in self.agents]
            _, rewards, done, truncated, info = self._process_actions(actions_list)

            # Update agent states
            for a in self.agents:
                self.rewards[a] = rewards[self.agent_name_mapping[a]]
                self._cumulative_rewards[a] = rewards[self.agent_name_mapping[a]]
                self.terminations[a] = done
                self.truncations[a] = truncated
                self.infos[a] = info
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()

    def _was_dead_step(self, action):
        """Handle step for an agent that is already terminated or truncated."""
        # When all agents are done, clear everything
        if self._agent_selector.is_last():
            self.agents = []
            self.rewards = {}
            self.terminations = {}
            self.truncations = {}
            self.infos = {}
            self._cumulative_rewards = {}

        self.agent_selection = self._agent_selector.next()

    def _clear_rewards(self):
        """Clear the rewards dictionary."""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        """Accumulate rewards into cumulative rewards (deprecated, not used in this implementation)."""
        for agent in self.rewards:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def _validate_actions(self, actions):
        """Validate and sanitize actions."""
        # sanitize actions
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    f"{player.name}{player.position} attempted invalid action {action}."
                )
                actions[i] = Action.NONE
        return actions

    def _get_proposed_position(self, player, action):
        if action == Action.NORTH:
            return (player.position[0] - 1, player.position[1])
        elif action == Action.SOUTH:
            return (player.position[0] + 1, player.position[1])
        elif action == Action.WEST:
            return (player.position[0], player.position[1] - 1)
        elif action == Action.EAST:
            return (player.position[0], player.position[1] + 1)
        return player.position

    def _resolve_player_movements(self, actions):
        """Resolve player movements and identify loading players."""
        loading_players = set()
        collisions = defaultdict(list)

        # check for collisions
        for player, action in zip(self.players, actions):
            if player.position is None:
                continue

            target_pos = self._get_proposed_position(player, action)
            collisions[target_pos].append(player)

            if action == Action.LOAD:
                loading_players.add(player)

        # do movements for non-colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # collision
                self._step_collisions += 1
                continue
            v[0].position = k

        return loading_players

    def _process_food_loading(self, loading_players):
        """Process food loading attempt by players."""
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            if player.position is None:
                continue
            food_location = self.adjacent_food_location(*player.position)
            if food_location is None:
                # No adjacent food, skip this player
                continue
            frow, fcol = food_location

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            # remove other participants from the set so we don't process them again
            loading_players = loading_players - set(adj_players)

            # the food was loaded and each player scores points
            self._step_foods_collected += 1
            if len(adj_players) > 1:
                self._step_cooperative_collections += 1
            else:
                self._step_solo_collections += 1

            energy_per_player = self.food_energy_value / len(adj_players)
            for a in adj_players:
                # Sustainability reward: less reward if already full
                hunger_ratio = 1.0 - (a.energy / a.max_energy)
                a.reward = float(energy_per_player * hunger_ratio)
                
                # Restore energy
                a.energy = min(a.max_energy, a.energy + energy_per_player)

            # and the food is removed
            self.field[frow, fcol] = 0

    def _update_game_state(self):
        """Update game over state and player scores."""
        any_dead = False
        for p in self.players:
            if p.is_dead:
                p.position = None
                any_dead = True

        self._game_over = (
            any_dead or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            if p.score is not None:
                p.score += p.reward
            else:
                p.score = p.reward

    def _process_actions(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0
            if not p.is_dead:
                p.energy -= self.energy_depletion_rate

        # Reset per-step metric counters
        self._step_foods_collected = 0
        self._step_cooperative_collections = 0
        self._step_solo_collections = 0
        self._step_failed_loads = 0
        self._step_collisions = 0

        actions = self._validate_actions(actions)

        # Track action distribution and apply movement energy cost
        self._step_action_counts = {}
        for p, a in zip(self.players, actions):
            name = a.name
            self._step_action_counts[name] = self._step_action_counts.get(name, 0) + 1
            
            # Moving takes extra energy
            if not p.is_dead and a in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]:
                p.energy -= self.energy_depletion_rate

        loading_players = self._resolve_player_movements(actions)
        self._process_food_loading(loading_players)
        
        # Logistic food regeneration (SFP Equation 11)
        # r_{t+1} = α * r_t - (α - 1) / K * r_t² - total_foraged
        # Uses continuous _food_level for accurate math, then syncs the grid.
        alpha = self.food_regeneration_rate  # replenishment rate (α > 1)
        K = float(self.max_num_food)         # carrying capacity
        r_t = self._food_level               # pre-foraging continuous level
        total_foraged = float(self._step_foods_collected)

        if K > 0 and r_t > 0:
            r_next = alpha * r_t - (alpha - 1.0) / K * (r_t ** 2) - total_foraged
        else:
            # Point of no return: if r_t == 0, no regrowth is possible
            r_next = -total_foraged

        r_next = max(0.0, min(r_next, K))  # clamp to [0, K]
        self._food_level = r_next

        # Sync grid to match the calculated food level
        current_grid_food = int(np.count_nonzero(self.field))
        target_grid_food = int(round(r_next))
        diff = target_grid_food - current_grid_food

        if diff > 0:
            self._spawn_food_units(diff)
        elif diff < 0:
            self._remove_food_units(-diff)

        self._update_game_state()

        rewards = [p.reward for p in self.players]
        done = self._game_over
        truncated = False
        info = self._get_info()

        return self._make_gym_obs(), rewards, done, truncated, info

    def _spawn_food_units(self, count):
        """Spawn `count` food units near existing food patches or food zones."""
        spawned = 0
        attempts = 0

        # Gather candidate centers: existing food locations + food zones
        food_positions = list(zip(*np.nonzero(self.field)))
        centers = food_positions + list(self.food_zones)
        if not centers:
            # If no food and no zones, fallback to random interior positions
            centers = [(self._np_random.integers(1, self.rows - 1),
                        self._np_random.integers(1, self.cols - 1))]

        while spawned < count and attempts < 1000:
            attempts += 1
            center = centers[self._np_random.integers(0, len(centers))]
            row = int(np.clip(center[0] + self._np_random.integers(-2, 3), 1, self.rows - 2))
            col = int(np.clip(center[1] + self._np_random.integers(-2, 3), 1, self.cols - 2))
            if self._is_empty_location(row, col):
                self.field[row, col] = 1
                spawned += 1

    def _remove_food_units(self, count):
        """Remove `count` food units randomly from the grid."""
        food_positions = list(zip(*np.nonzero(self.field)))
        if not food_positions:
            return
        self._np_random.shuffle(food_positions)
        for i in range(min(count, len(food_positions))):
            r, c = food_positions[i]
            self.field[r, c] = 0

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()

        if self.viewer is not None:
            return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()

    def test_make_gym_obs(self):
        """Test wrapper to test the current observation in a public manner."""
        return self._make_gym_obs()

    def test_gen_valid_moves(self):
        """Wrapper around a private method to test if the generated moves are valid."""
        try:
            self._gen_valid_moves()
        except Exception as _:
            return False
        return True
