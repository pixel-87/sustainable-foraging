from __future__ import annotations

from collections import defaultdict
from typing import Any

from pettingzoo.utils.wrappers import BaseWrapper


class ForagingMetricsWrapper(BaseWrapper):
    """AEC wrapper that accumulates step-level metrics into episode-level metrics."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._reset_metrics()
        self._last_cycle_foods = 0
        self._last_cycle_coop = 0
        self._last_cycle_solo = 0
        self._last_cycle_failed = 0
        self._last_cycle_collisions = 0
        self._last_cycle_action_counts = defaultdict(int)
        self._last_cycle_agent_rewards = defaultdict(float)

    def _reset_metrics(self) -> None:
        self._ep_reward = 0.0
        self._ep_length = 0
        self._foods = 0
        self._coop = 0
        self._solo = 0
        self._failed = 0
        self._collisions = 0
        self._action_counts: dict[str, int] = defaultdict(int)
        self._agent_rewards: dict[str, float] = defaultdict(float)
        self._food_remaining = 0
        self._step_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        super().reset(seed=seed, options=options)
        self._reset_metrics()

    def step(self, action: Any) -> None:
        agent = self.agent_selection
        was_done = self.terminations[agent] or self.truncations[agent]

        super().step(action)

        if not was_done:
            self._step_count += 1
            info = self.infos[agent]

            # AEC environment updates these accumulators during the cycle.
            # We only want to add the final values at the end of the cycle (when agent is the last one).
            agent_selector = getattr(self.unwrapped, "_agent_selector", None)
            if agent_selector and agent_selector.is_last():
                self._ep_length += 1
                self._foods += info.get("foods_collected", 0)
                self._coop += info.get("cooperative_collections", 0)
                self._solo += info.get("solo_collections", 0)
                self._failed += info.get("failed_loads", 0)
                self._collisions += info.get("collisions", 0)

                for act_name, count in info.get("action_counts", {}).items():
                    self._action_counts[act_name] += count

                for j, r in enumerate(info.get("per_agent_rewards", [])):
                    self._agent_rewards[f"agent_{j}"] += r

            self._food_remaining = info.get("food_remaining", 0)
            self._ep_reward += self.rewards[agent]

        # Attach metrics to info if any agent is done
        if any(self.terminations.values()) or any(self.truncations.values()):
            self.infos[agent]["episode_metrics"] = {
                "reward_total": self._ep_reward,
                "length": self._ep_length,
                "foods_collected": self._foods,
                "cooperative_collections": self._coop,
                "solo_collections": self._solo,
                "failed_loads": self._failed,
                "collisions": self._collisions,
                "food_remaining_end": self._food_remaining,
                "action_counts": dict(self._action_counts),
                "agent_rewards": dict(self._agent_rewards),
            }
            # We don't reset metrics here; they reset on next `reset()`.
