import sys

from pettingzoo.test import api_test
from sustainable_foraging.foraging.aecEnvironment import ForagingEnv


def main():
    try:
        env = ForagingEnv(
            players=2,
            max_energy=50,
            food_energy_value=10,
            energy_depletion_rate=1,
            food_regeneration_rate=1.5,
            num_food_zones=2,
            observe_agent_energy=True,
            field_size=(8, 8),
            max_num_food=2,
            sight=8,
            max_episode_steps=50,
            normalize_reward=True,
            grid_observation=False,
            penalty=0.0,
        )
        api_test(env, num_cycles=100, verbose_progress=True)
        print("API test passed!")
    except Exception as e:
        print(f"API test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
