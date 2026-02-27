# Sustainable Benchmark Protocol

This project uses a fixed benchmark protocol for fair algorithm comparisons.

## Benchmark Identity

- Name: `sustainable_v1`
- Presets: `easy`, `fair`, `hard`
- Source of truth: `lbforaging/foraging/sustainable_benchmark.py`

## Fixed Environment Presets

`easy`
- `players=2`, `max_energy=120`, `food_energy_value=12`
- `energy_depletion_rate=1`, `food_regeneration_rate=0.20`, `num_food_zones=3`
- `field_size=(8,8)`, `max_num_food=3`, `sight=8`, `max_episode_steps=60`

`fair`
- `players=2`, `max_energy=100`, `food_energy_value=10`
- `energy_depletion_rate=1`, `food_regeneration_rate=0.10`, `num_food_zones=2`
- `field_size=(8,8)`, `max_num_food=2`, `sight=8`, `max_episode_steps=50`

`hard`
- `players=2`, `max_energy=80`, `food_energy_value=8`
- `energy_depletion_rate=2`, `food_regeneration_rate=0.03`, `num_food_zones=1`
- `field_size=(8,8)`, `max_num_food=2`, `sight=8`, `max_episode_steps=50`

## Fixed Training Defaults

- `total_timesteps=200000`
- `learning_rate=0.001`
- `num_envs=8`
- `batch_size=2048`
- `eval_episodes=20`

These defaults are set in `BENCHMARK_TRAINING_DEFAULTS` and used by `scripts/train_sb3.py`.

## Fixed Seed Splits

- Train seeds: `11, 22, 33, 44, 55`
- Eval seeds: `101, 202, 303, 404, 505`

These are defined in `BENCHMARK_SEEDS`.

## Reproducibility Rules

- Do not change preset parameters between algorithm runs.
- Do not change timesteps, batch size, or seed split for direct comparisons.
- Always keep `logs/<run>/config.json` and compare runs with the same preset.

## Useful Commands

- Show presets:
  - `python -m scripts.train_sb3 --list-presets`
- Show benchmark defaults/seeds:
  - `python -m scripts.train_sb3 --show-benchmark-settings`
- Train with fixed defaults:
  - `python -m scripts.train_sb3 --preset fair`
