from __future__ import annotations

from typing import Any

import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit.vector.constructors import MakeCPUAsyncConstructor
from supersuit.vector.vector_constructors import vec_env_args
from sustainable_foraging.foraging import AECForagingEnv
from sustainable_foraging.foraging.sustainable_benchmark import get_preset

from .log_utils import ForagingMetricsWrapper


def custom_concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class="gymnasium"):
    """Fix for supersuit's MakeCPUAsyncConstructor bug missing obs_space and act_space args."""
    num_cpus = min(num_cpus, num_vec_envs)
    if num_cpus > 1:
        constructor = MakeCPUAsyncConstructor(num_cpus)
        args = vec_env_args(vec_env, num_vec_envs)
        obs_space = vec_env.observation_space
        act_space = vec_env.action_space
        vec_env = constructor(args[0], obs_space, act_space)
    else:
        vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class="gymnasium")

    if base_class == "gymnasium":
        return vec_env
    elif base_class == "stable_baselines":
        from supersuit.vector.sb_vector_wrapper import SBVecEnvWrapper
        return SBVecEnvWrapper(vec_env)
    elif base_class == "stable_baselines3":
        from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
        return SB3VecEnvWrapper(vec_env)
    else:
        raise ValueError("base_class unsupported")

def make_env(
    preset: str = "fair",
    num_envs: int = 1,
    config_overrides: dict[str, Any] | None = None,
    vectorize_for_cleanrl_sb3: bool = False,
    base_class: str = "gymnasium",
    num_cpus: int = 0,
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
        env = custom_concat_vec_envs_v1(
            env,
            num_vec_envs=num_envs,
            num_cpus=num_cpus,
            base_class=base_class,
        )

    return env, env_config
