"""Formal benchmark presets for sustainable foraging experiments.

These presets define the task, not the training hyperparameters.
Use the same preset across algorithms/libraries for fair comparisons.
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

SUSTAINABLE_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "players": 2,
        "max_energy": 120,
        "food_energy_value": 12,
        "energy_depletion_rate": 1,
        "food_regeneration_rate": 2.0,  # α: fast logistic regrowth
        "num_food_zones": 3,
        "field_size": (8, 8),
        "max_num_food": 3,
        "sight": 8,
        "max_episode_steps": 60,
        "grid_observation": True,
    },
    "fair": {
        "players": 2,
        "max_energy": 100,
        "food_energy_value": 10,
        "energy_depletion_rate": 1,
        "food_regeneration_rate": 1.5,  # α: moderate logistic regrowth
        "num_food_zones": 2,
        "field_size": (8, 8),
        "max_num_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "grid_observation": True,
    },
    "hard": {
        "players": 2,
        "max_energy": 80,
        "food_energy_value": 8,
        "energy_depletion_rate": 2,
        "food_regeneration_rate": 1.1,  # α: barely above replacement
        "num_food_zones": 1,
        "field_size": (8, 8),
        "max_num_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "grid_observation": True,
    },
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
