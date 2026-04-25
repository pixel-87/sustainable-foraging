# Sustainable Benchmark Protocol

This project uses a fixed benchmark protocol for fair algorithm comparisons.

## Benchmark Identity

- Name: `sustainable_v1`
- Presets: `easy`, `fair`, `hard`
- Source of truth: `sustainable_foraging/foraging/sustainable_benchmark.py`

## Critical Alpha Threshold

Each preset's replenishment rate (α) is computed from the **SFP logistic growth sustainability condition**:

```
α_critical = 1 + (4 · N · d · c) / (K · food_energy_value)
```

| Symbol | Meaning |
|--------|---------|
| N | Number of agents |
| d | `energy_depletion_rate` |
| c | Cost multiplier (1=still, 2=always moving) |
| K | `max_num_food` (carrying capacity) |

At `α = α_critical`, the environment can only sustain agents that act near-optimally.
Difficulty is controlled by the cost multiplier `c` assumption.

## Fixed Environment Presets

| Preset | α | c | N | d | K | E | Steps |
|--------|------|-----|---|---|---|----|----|
| `easy` | 1.2222 | 1.0 | 2 | 1 | 3 | 12 | 500 |
| `fair` | 1.6000 | 1.5 | 2 | 1 | 2 | 10 | 500 |
| `hard` | 3.0000 | 2.0 | 2 | 2 | 2 | 8 | 500 |

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
