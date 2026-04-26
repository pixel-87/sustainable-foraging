from __future__ import annotations

from typing import Any

import supersuit as ss
from pettingzoo.utils import aec_to_parallel

from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import get_preset
from scripts.core.log_utils import ForagingMetricsWrapper


def make_env(
    preset: str = "fair",
    num_envs: int = 1,
    config_overrides: dict[str, Any] | None = None,
    vectorize_for_cleanrl_sb3: bool = False,
    base_class: str = "gymnasium",
) -> tuple[Any, dict[str, Any]]:
    """Unified environment factory for all training scripts.

    Returns:
        tuple[env, env_config]
    """
    env_config = get_preset(preset)
    if config_overrides:
        env_config.update(config_overrides)

    # 1. Base AEC environment
    env = AECForagingEnv(**env_config)

    # 2. Metric accumulator wrapper (AEC level)
    env = ForagingMetricsWrapper(env)

    # 3. Convert to parallel environment (required for almost all algorithms)
    env = aec_to_parallel(env)

    if vectorize_for_cleanrl_sb3:
        # Supersuit vectorization for SB3 and CleanRL
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env,
            num_vec_envs=num_envs,
            num_cpus=0,
            base_class=base_class,
        )

    return env, env_config
