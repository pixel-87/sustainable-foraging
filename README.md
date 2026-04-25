# Sustainable Foraging Benchmark

A reproducible benchmark for comparing multi-agent RL algorithms on the Sustainable Foraging environment (PettingZoo AEC API).

Forked from [lb-foraging](https://github.com/semitable/lb-foraging). Built on the Level-Based Foraging framework, adapted for sustainable foraging research.

<p align="center">
  <img width="450px" src="docs/img/lbf.gif" align="center" alt="Sustainable Foraging" />
</p>

## Install

```sh
git clone https://github.com/pixel-87/sustainable-foraging.git
cd sustainable-foraging
uv sync
```

## Quick Start

```sh
# Train with stable-baselines3 (choose preset: easy, fair, hard)
python -m scripts.train_sb3 --preset fair

# Run inference with a trained model
python -m scripts.inference_sb3 --preset fair --model logs/<run_name>/model

# List available presets and settings
python -m scripts.train_sb3 --list-presets
```

The benchmark protocol is documented in `docs/benchmark_protocol.md`.

## Creating Environments

This is a PettingZoo AEC environment:

```python
from sustainable_foraging.foraging import AECForagingEnv

env = AECForagingEnv(
    players=2,
    field_size=(8, 8),
    max_num_food=2,
    sight=8,
    max_episode_steps=500,
)

env.reset()
while env.agents:
    agent = env.agent_selection
    obs, reward, terminated, truncated, info = env.last()
    action = env.action_space(agent).sample()
    env.step(action)
    env.render()
```

## Citation

If you use this benchmark, please cite the original work:

```bibtex
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Schäfer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

```bibtex
@inproceedings{papoudakis2021benchmarking,
  title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
  author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
  year={2021}
}
```

## Contributing

Contributions are welcome! Please open an issue to discuss changes before submitting PRs.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
