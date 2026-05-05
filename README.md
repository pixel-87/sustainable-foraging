# Sustainable Foraging Benchmark

A reproducible benchmark for comparing multi-agent RL algorithms on the Sustainable Foraging environment. This environment strictly conforms to the PettingZoo Agent-Environment Cycle (AEC) API, ensuring deterministic state transitions and sequential execution.

Forked from [lb-foraging](https://github.com/semitable/lb-foraging), but completely overhauled to incorporate logistic resource regeneration and logarithmic reward shaping to simulate the "Tragedy of the Commons."

## The Sustainable Foraging Problem

The Sustainable Foraging Problem (SFP) is a multi-agent social dilemma that models Common-Pool Resource (CPR) management. In this grid-world environment, independent agents navigate and forage for resources. Crucially, the resources (food) regenerate according to a **logistic growth function**. 

If agents forage sustainably, the environment remains abundant, yielding long-term rewards for all. However, if the joint foraging rate exceeds the Maximum Sustainable Yield (the peak of the logistic regrowth parabola), the ecosystem crosses a "Point of No Return" and collapses, resulting in inevitable starvation. This creates a "Wicked Problem" where short-term individual reward maximisation is directly at odds with long-term collective survival.

## Aims and Objectives

The primary aim of this project is to implement and benchmark a MARL environment that accurately models the "Tragedy of the Commons." This is supported by the following measurable objectives:

1. **Architectural Robustness**: Achieve 100% compliance with the PettingZoo AEC API to enforce sequential stepping and atomic actions, eliminating the race conditions inherent in legacy simultaneous-action APIs.
2. **Mathematical Stability**: Implement a logistic resource regeneration model and logarithmic reward scaling that maintains a "knife-edge" equilibrium, providing a stable micro-ecosystem for evaluating restraint.
3. **Benchmarking Rigour**: Evaluate distinct MARL algorithms (Independent Learners, Centralised Policy Gradients, and Value-Decomposition architectures) using a standardised multi-seed evaluation protocol.
4. **Partial Observability**: Evaluate algorithms under limited visibility (POMDP) to test the impact of thermodynamic exploration costs and global state knowledge on long-term sustainability.

## Installation & Usage

### Installation

You can install the environment directly via pip:

```bash
pip install sustainable-foraging
```

Or build from source using your preferred package manager:

```bash
git clone https://github.com/pixel-87/sustainable-foraging.git
cd sustainable-foraging
pip install -e .
```

#### Using uv (modern Python package manager)

```bash
uv sync
```

#### Using Nix

Nix handles the dependency environment automatically, so no pip install needed. Just run `nix develop` (or `direnv allow` for automatic activation). The project uses [uv2nix](https://github.com/adis-blomer/uv2nix) to generate Nix derivations from `pyproject.toml` and `uv.lock`.

```bash
nix develop
# or
direnv allow
```

Otherwise, see the sections above for `pip` or `uv` installation.

### Environment Interface

The environment conforms to the PettingZoo Agent-Environment Cycle (AEC) model. 

```python
from sustainable_foraging.foraging import AECForagingEnv

# 1. Instantiate the environment
env = AECForagingEnv(
    players=2,
    field_size=(8, 8),
    max_num_food=2,
    sight=8,
    max_episode_steps=500,
)

# 2. Start an episode
env.reset()

# 3. Iterate sequentially through agents
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    
    if terminated or truncated:
        action = None
    else:
        # Sample or predict an action
        action = env.action_space(agent).sample()
        
    env.step(action)
    env.render()

env.close()
```

### Running Benchmarks & Algorithms

The repository includes a unified entry point (`scripts/train.py`) to easily train various algorithms across different RL libraries (CleanRL, Stable-Baselines3, RLlib). 

For example, to train QMIX using CleanRL on the "fair" (knife-edge) preset for 3 million timesteps:

```bash
python scripts/train.py \
    --library cleanrl \
    --algorithm qmix \
    --preset fair \
    --timesteps 3000000 \
    --num-envs 4 \
    --num-cpus 4 \
    --name qmix_demo_run \
    --lr 0.0001 \
    --exploration-fraction 0.5
```

You can view all available algorithms and hyperparameter options via:
```bash
python scripts/train.py --help
```

### Integration with Stable-Baselines3

The environment can be easily parallelized for training using PettingZoo's SuperSuit wrappers:

```python
from sustainable_foraging.foraging import AECForagingEnv
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
import supersuit as ss
from stable_baselines3 import PPO

env = AECForagingEnv(players=2, field_size=(8, 8), max_num_food=4)
env = aec_to_parallel(env)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = SB3VecEnvWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

## Benchmark Results

A comprehensive benchmark was conducted across 6 MARL algorithms in a "knife-edge" equilibrium environment. 

### Key Findings:
- **Independent Learners & Centralised Policy Gradients (PPO, MAPPO, A2C, DQN)**: Broadly succumb to the Tragedy of the Commons. Because they update based heavily on individual returns or lack coordinated memory, they rapidly over-forage and trigger ecological collapse.
- **Value-Decomposition Methods (VDN, QMIX)**: Successfully solve the dilemma. By optimising for a decomposed global team reward, greedy behaviour is rendered mathematically detrimental. These agents learn to delay gratification and establish sustainable foraging rhythms.

#### Fully Observable Environment Results (3 Seeds)
| Algorithm | Episode Length | Sustainability (Food Remaining) | Episodic Reward |
|-----------|----------------|----------------------------------|-----------------|
| CLEANRL/VDN | 101.09 ± 11.33 | 1.75 ± 0.15 | 95.79 ± 40.91 |
| CLEANRL/QMIX | 100.13 ± 8.24 | 1.86 ± 0.10 | 57.23 ± 33.74 |
| SB3/A2C | 83.12 ± 4.67 | 0.24 ± 0.14 | 11.47 ± 3.35 |
| CLEANRL/DQN | 70.10 ± 5.22 | 1.53 ± 0.22 | 49.21 ± 16.71 |
| BASELINE/GREEDY | 68.50 ± 0.34 | 0.42 ± 0.03 | 2.87 ± 0.09 |
| SB3/PPO | 59.95 ± 1.27 | 0.82 ± 0.30 | 36.97 ± 5.20 |
| CLEANRL/MAPPO | 58.61 ± 1.39 | 0.96 ± 0.32 | 29.05 ± 5.82 |

## References

If you build upon this environment or research, please consider referencing the foundational theoretical works and original frameworks:

1. **The Sustainable Foraging Problem (Theoretical Framework):**
   > Aishwaryaprajna and P. R. Lewis, “The Sustainable Foraging Problem,” in *2023 IEEE International Conference on Autonomic Computing and Self-Organizing Systems Companion (ACSOS-C)*, 2023.

2. **Common-Pool Resources (Economics & Social Science):**
   > E. Ostrom, R. Gardner, and J. Walker, *Rules, Games, and Common-pool Resources*. University of Michigan Press, 1994.

3. **Level-Based Foraging (Original Grid-World Mechanics):**
   > F. Christianos, L. Schäfer, and S. V. Albrecht, “Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

4. **PettingZoo API:**
   > J. K. Terry et al., “PettingZoo: Gym for Multi-Agent Reinforcement Learning.” arXiv, 2020.

## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

pixel-87

Project Link: [https://github.com/pixel-87/sustainable-foraging](https://github.com/pixel-87/sustainable-foraging)

Original Project: [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging)

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.