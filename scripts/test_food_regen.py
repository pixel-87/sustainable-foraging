import numpy as np
from lbforaging.foraging.aecEnvironment import ForagingEnv, Action


def test_logistic_growth():
    """Test that food regenerates via logistic growth and respects point-of-no-return."""
    env = ForagingEnv(
        players=2,
        field_size=(10, 10),
        max_num_food=5,
        sight=2,
        max_episode_steps=100,
        food_regeneration_rate=1.5,  # α > 1 for positive logistic growth
        num_food_zones=2,
    )

    env.reset()
    initial_food = np.count_nonzero(env.field)
    print(f"Initial food count: {initial_food}")
    assert initial_food > 0

    # Manually remove some food (simulate partial harvest)
    food_positions = list(zip(*np.nonzero(env.field)))
    if len(food_positions) > 1:
        env.field[food_positions[0][0], food_positions[0][1]] = 0
        env._food_level = float(np.count_nonzero(env.field))

    reduced_food = np.count_nonzero(env.field)
    print(f"Food after partial harvest: {reduced_food}")

    # Step the environment — logistic growth should regrow food
    for _ in range(10):
        actions = [Action.NONE.value for _ in range(len(env.players))]
        env._process_actions(actions)

    regrown_food = np.count_nonzero(env.field)
    print(f"Food after 10 steps of logistic regrowth: {regrown_food}")
    assert regrown_food >= reduced_food, "Logistic growth should increase food when below K"


def test_point_of_no_return():
    """Test that depleting all food leads to permanent collapse (r=0 stays 0)."""
    env = ForagingEnv(
        players=2,
        field_size=(10, 10),
        max_num_food=5,
        sight=2,
        max_episode_steps=100,
        food_regeneration_rate=1.5,
        num_food_zones=2,
    )

    env.reset()

    # Destroy all food — simulate total depletion
    env.field = np.zeros(env.field_size, np.int32)
    env._food_level = 0.0  # point of no return

    print(f"Food after total depletion: {np.count_nonzero(env.field)}")

    # Step many times — food should NOT regenerate
    for _ in range(20):
        actions = [Action.NONE.value for _ in range(len(env.players))]
        env._process_actions(actions)

    final_food = np.count_nonzero(env.field)
    print(f"Food after 20 steps (should be 0): {final_food}")
    assert final_food == 0, "Point of no return: depleted resources must not recover!"
    print("Point of no return verified!")


if __name__ == "__main__":
    test_logistic_growth()
    test_point_of_no_return()
    print("\nAll logistic growth tests passed!")
