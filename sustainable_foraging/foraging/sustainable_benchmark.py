"""Formal benchmark presets for sustainable foraging experiments.

These presets define the task, not the training hyperparameters.
Use the same preset across algorithms/libraries for fair comparisons.

    The critical a (food_regeneration_rate) is derived from the SFP logistic
    growth equation so that sustainability is only possible with near-perfect
    agent behaviour.  The formula is:

    a_critical = 1 + (4 · N · d · c) / (K · food_energy_value)

where:
    N = number of agents
    d = energy_depletion_rate
    c = average energy cost multiplier per step
        (1 = agent never moves, 2 = agent moves every step)
    K = max_num_food  (carrying capacity)
    food_energy_value = energy gained per food unit
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

BENCHMARK_NAME = "sustainable_v1"

# Fixed benchmark protocol for reproducible algorithm comparisons.
BENCHMARK_TRAINING_DEFAULTS: dict[str, Any] = {
    "total_timesteps": 200_000,
    "learning_rate": 1e-3,
    "num_envs": 8,
    "batch_size": 2048,
    "eval_episodes": 20,
}

BENCHMARK_SEEDS: dict[str, tuple[int, ...]] = {
    "train": (11, 22, 33, 44, 55),
    "eval": (101, 202, 303, 404, 505),
}


def compute_critical_alpha(
    players: int,
    energy_depletion_rate: int | float,
    max_num_food: int,
    food_energy_value: int | float,
    cost_multiplier: float = 1.5,
) -> float:
    """Compute the critical replenishment rate a for the SFP logistic model.

    At a = a_critical the maximum possible logistic regrowth exactly equals
    the minimum harvest rate the agents need to survive.  Any a below this
    makes collapse inevitable; any a above gives a margin of safety.

    Parameters
    ----------
    players : int
        Number of agents (N).
    energy_depletion_rate : int | float
        Base energy lost per step (d).
    max_num_food : int
        Carrying capacity of the grid (K).
    food_energy_value : int | float
        Energy restored per food unit consumed.
    cost_multiplier : float
        Average per-step energy cost as a multiple of d.
        1.0 = agent never moves (base cost only).
        2.0 = agent moves every step (base + movement).
        Default 1.5 is a realistic middle ground.

    Returns
    -------
    float
        The critical a value (always > 1).
    """
    N = players
    d = energy_depletion_rate
    K = max_num_food
    E = food_energy_value
    c = cost_multiplier

    # Minimum food units consumed per step for all agents to survive
    F_min = (N * d * c) / E

    # At the logistic inflection point (r* = K/2), maximum growth = (a-1)·K/4
    # Setting (a-1)·K/4 = F_min and solving for a:
    alpha_critical = 1.0 + (4.0 * F_min) / K

    return alpha_critical


# ---------------------------------------------------------------------------
# Presets – α is set to the critical threshold so that only near-perfect
# agents can sustain the environment.  Difficulty comes from the cost
# multiplier assumption:
#   easy  → c=1.0  (generous: assumes minimal movement cost)
#   fair  → c=1.5  (realistic: moderate movement)
#   hard  → c=2.0  (strict: assumes agents move every step)
# ---------------------------------------------------------------------------

_EASY_PARAMS: dict[str, Any] = {
    "players": 2,
    "max_energy": 120,
    "food_energy_value": 12,
    "energy_depletion_rate": 1,
    "num_food_zones": 3,
    "field_size": (8, 8),
    "max_num_food": 3,
    "sight": 8,
    "max_episode_steps": 500,
    "grid_observation": True,
}

_FAIR_PARAMS: dict[str, Any] = {
    "players": 2,
    "max_energy": 100,
    "food_energy_value": 10,
    "energy_depletion_rate": 1,
    "num_food_zones": 2,
    "field_size": (8, 8),
    "max_num_food": 2,
    "sight": 2,
    "max_episode_steps": 500,
    "grid_observation": True,
}

_HARD_PARAMS: dict[str, Any] = {
    "players": 2,
    "max_energy": 80,
    "food_energy_value": 8,
    "energy_depletion_rate": 2,
    "num_food_zones": 1,
    "field_size": (8, 8),
    "max_num_food": 2,
    "sight": 8,
    "max_episode_steps": 500,
    "grid_observation": True,
}

# Compute α_critical for each preset
for _params, _c in [(_EASY_PARAMS, 1.0), (_FAIR_PARAMS, 1.5), (_HARD_PARAMS, 2.0)]:
    _params["food_regeneration_rate"] = round(
        compute_critical_alpha(
            players=_params["players"],
            energy_depletion_rate=_params["energy_depletion_rate"],
            max_num_food=_params["max_num_food"],
            food_energy_value=_params["food_energy_value"],
            cost_multiplier=_c,
        ),
        4,
    )

SUSTAINABLE_PRESETS: dict[str, dict[str, Any]] = {
    "easy": _EASY_PARAMS,
    "fair": _FAIR_PARAMS,
    "hard": _HARD_PARAMS,
}


def list_presets() -> tuple[str, ...]:
    return tuple(SUSTAINABLE_PRESETS.keys())


def get_preset(name: str) -> dict[str, Any]:
    if name not in SUSTAINABLE_PRESETS:
        valid = ", ".join(list_presets())
        raise ValueError(f"Unknown preset '{name}'. Choose one of: {valid}")
    return deepcopy(SUSTAINABLE_PRESETS[name])


def get_training_defaults() -> dict[str, Any]:
    return deepcopy(BENCHMARK_TRAINING_DEFAULTS)


def get_seeds(split: str) -> tuple[int, ...]:
    if split not in BENCHMARK_SEEDS:
        valid = ", ".join(BENCHMARK_SEEDS.keys())
        raise ValueError(f"Unknown seed split '{split}'. Choose one of: {valid}")
    return BENCHMARK_SEEDS[split]
