from lbforaging.foraging.environment import ForagingEnv as GymForagingEnv  # noqa
from lbforaging.foraging.aecEnvironment import ForagingEnv as AECForagingEnv  # noqa
from lbforaging.foraging.sustainable_benchmark import (  # noqa
    BENCHMARK_NAME,
    BENCHMARK_SEEDS,
    BENCHMARK_TRAINING_DEFAULTS,
    SUSTAINABLE_PRESETS,
    get_preset,
    get_seeds,
    get_training_defaults,
    list_presets,
)

# Keep the Gym-style `ForagingEnv` as the package default so `gym.make` works.
ForagingEnv = GymForagingEnv

# If you want the PettingZoo AEC implementation, import `AECForagingEnv`.
