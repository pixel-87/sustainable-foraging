import numpy as np
from lbforaging.foraging.aecEnvironment import ForagingEnv, Action


def test_food_regeneration():
    # Instantiate the environment
    env = ForagingEnv(
        players=2,
        field_size=(10, 10),
        max_num_food=5,
        sight=2,
        max_episode_steps=100,
        food_regeneration_rate=1.0,  # Set to 1.0 to guarantee regeneration attempt
        num_food_zones=2,
    )

    env.reset()

    # Manually remove all food
    env.field = np.zeros(env.field_size, np.int32)

    initial_food_count = np.count_nonzero(env.field)
    print(f"Initial food count after removal: {initial_food_count}")
    assert initial_food_count == 0

    # Step the environment many times
    for _ in range(20):
        # Provide NONE actions for all players
        actions = [Action.NONE for _ in range(len(env.players))]
        env._process_actions(actions)

    final_food_count = np.count_nonzero(env.field)
    print(f"Final food count after stepping: {final_food_count}")

    # Assert that food regenerates
    assert final_food_count > 0, "Food did not regenerate!"
    print("Food regeneration successful!")


if __name__ == "__main__":
    test_food_regeneration()
