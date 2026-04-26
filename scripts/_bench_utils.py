"""Shared utilities for sustainable foraging benchmark training scripts.

Provides a consistent metrics tracker and standard CLI arg parser so that all
algorithms (SB3, RLlib, CleanRL) report identical CSV files consumed by
``scripts/compare_algorithms.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from sustainable_foraging.foraging.sustainable_benchmark import (
    BENCHMARK_NAME,
    BENCHMARK_SEEDS,
    get_preset,
    get_training_defaults,
    list_presets,
)


class MetricsTracker:
    """Tracks per-episode metrics and writes them to a consistent CSV format.

    Designed to work with *any* RL library.  The training loop calls
    :meth:`on_episode_end` with the accumulated stats for a finished episode.
    """

    CSV_COLUMNS: list[str] = [
        "episode",
        "timestep",
        "wall_time",
        # Episode-level
        "reward_total",
        "length",
        # Food
        "foods_collected",
        "cooperative_collections",
        "solo_collections",
        "failed_loads",
        "food_remaining_end",
        # Collisions
        "collisions",
        # Action distribution
        "actions_north",
        "actions_south",
        "actions_east",
        "actions_west",
        "actions_load",
        "actions_none",
        # Per-agent rewards
        "agent_rewards",
    ]

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path: Path = Path(csv_path)
        self._csv_file: TextIOWrapper = open(self.csv_path, "w", newline="")
        self._csv_writer: Any = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_COLUMNS)
        self._ep_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_episode_end(self, timestep: int, ep_stats: dict[str, Any]) -> None:
        """Write one finished episode to CSV."""
        self._ep_count += 1

        ep_reward: float = float(ep_stats.get("reward_total", 0.0))
        ep_length: int = int(ep_stats.get("length", 0))
        foods: int = int(ep_stats.get("foods_collected", 0))
        coop: int = int(ep_stats.get("cooperative_collections", 0))
        solo: int = int(ep_stats.get("solo_collections", 0))
        failed: int = int(ep_stats.get("failed_loads", 0))
        food_remaining: int = int(ep_stats.get("food_remaining_end", 0))
        collisions: int = int(ep_stats.get("collisions", 0))

        ac: dict[str, int] = ep_stats.get("action_counts", {})
        agent_rewards: dict[str, float] = ep_stats.get("agent_rewards", {})

        self._csv_writer.writerow(
            [
                self._ep_count,
                timestep,
                time.time(),
                round(ep_reward, 6),
                ep_length,
                foods,
                coop,
                solo,
                failed,
                food_remaining,
                collisions,
                ac.get("NORTH", 0),
                ac.get("SOUTH", 0),
                ac.get("EAST", 0),
                ac.get("WEST", 0),
                ac.get("LOAD", 0),
                ac.get("NONE", 0),
                json.dumps(agent_rewards),
            ]
        )
        self._csv_file.flush()

    @property
    def episode_count(self) -> int:
        return self._ep_count

    def close(self) -> None:
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()
            print(f"  Metrics saved to: {self.csv_path}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def save_experiment_config(
    log_dir: Path,
    run_name: str,
    preset: str,
    algorithm: str,
    library: str,
    total_timesteps: int,
    **extra_hyperparams: Any,
) -> Path:
    """Persist an experiment config that ``compare_algorithms.py`` can load."""
    env_config: dict[str, Any] = get_preset(preset)

    experiment_config: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "preset": preset,
        "run_name": run_name,
        "total_timesteps": total_timesteps,
        "algorithm": algorithm,
        "library": library,
        "seed_splits": BENCHMARK_SEEDS,
        "environment": env_config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra_hyperparams,
    }

    config_path: Path = log_dir / "config.json"
    serializable: Any = json.loads(
        json.dumps(
            experiment_config,
            default=lambda o: list(o) if isinstance(o, tuple) else str(o),
        )
    )
    with open(config_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return config_path


# ---------------------------------------------------------------------------
# Standard CLI
# ---------------------------------------------------------------------------


def get_standard_parser(
    description: str = "Train agent on Sustainable Foraging",
) -> argparse.ArgumentParser:
    """Return the standard argument parser shared by all benchmark scripts."""
    defaults: dict[str, Any] = get_training_defaults()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=defaults["total_timesteps"],
        help=f"Total training timesteps (default: {defaults['total_timesteps']})",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Run name (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=defaults["learning_rate"],
        help=f"Learning rate (default: {defaults['learning_rate']})",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fair",
        choices=list_presets(),
        help="Sustainable benchmark preset (default: fair)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=defaults["num_envs"],
        help=f"Number of vectorized environments (default: {defaults['num_envs']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults["batch_size"],
        help=f"Batch size (default: {defaults['batch_size']})",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=0,
        help="Number of CPUs for vectorized environments (0=sequential) (default: 0)",
    )
    return parser
