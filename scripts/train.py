#!/usr/bin/env python3
"""Unified entry point for Sustainable Foraging benchmarking."""

import argparse
import sys
from typing import Any

from sustainable_foraging.foraging.sustainable_benchmark import BENCHMARK_NAME, get_training_defaults, list_presets
from scripts._bench_utils import get_standard_parser


def parse_args() -> argparse.Namespace:
    parser = get_standard_parser(description="Train agent on Sustainable Foraging (Unified Entry)")
    
    # Core dispatch arguments
    parser.add_argument(
        "--library",
        type=str,
        choices=["sb3", "cleanrl", "rllib"],
        required=True,
        help="Which RL library to use",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "a2c", "dqn", "mappo", "sac", "vdn", "qmix"],
        required=True,
        help="Which RL algorithm to use",
    )
    
    # Add CleanRL specific args
    parser.add_argument("--seed", type=int, default=1, help="Random seed (CleanRL)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available (CleanRL)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    
    # MAPPO specifics
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout length per env (CleanRL MAPPO)")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (CleanRL MAPPO)")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of minibatches (CleanRL MAPPO)")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs (CleanRL MAPPO)")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coef (CleanRL MAPPO)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coef (CleanRL MAPPO)")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value func coef (CleanRL MAPPO)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm (CleanRL MAPPO)")
    parser.add_argument("--no-anneal-lr", action="store_false", dest="anneal_lr", help="Disable LR annealing (CleanRL MAPPO)")
    parser.set_defaults(anneal_lr=True)
    
    # DQN specifics
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size (CleanRL DQN)")
    parser.add_argument("--tau", type=float, default=1.0, help="Target network update rate (CleanRL DQN)")
    parser.add_argument("--target-network-frequency", type=int, default=500, help="Steps between target updates (CleanRL DQN)")
    parser.add_argument("--start-e", type=float, default=1.0, help="Start epsilon (CleanRL DQN)")
    parser.add_argument("--end-e", type=float, default=0.05, help="End epsilon (CleanRL DQN)")
    parser.add_argument("--exploration-fraction", type=float, default=0.5, help="Fraction of total steps for epsilon decay (CleanRL DQN)")
    parser.add_argument("--learning-starts", type=int, default=1000, help="Steps before training begins (CleanRL DQN)")
    parser.add_argument("--train-frequency", type=int, default=4, help="Steps between training updates (CleanRL DQN)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validation
    if args.algorithm == "sac":
        raise NotImplementedError("SAC not implemented yet.")
        
    if args.library == "sb3":
        if args.algorithm not in ["ppo", "a2c"]:
            raise ValueError(f"Algorithm {args.algorithm} not supported by sb3 runner (supported: ppo, a2c).")
        from scripts.core.runners.run_sb3 import run_sb3
        run_sb3(args, algorithm=args.algorithm)
        
    elif args.library == "cleanrl":
        if args.algorithm not in ["mappo", "dqn", "vdn", "qmix"]:
            raise ValueError(f"Algorithm {args.algorithm} not supported by cleanrl runner (supported: mappo, dqn, vdn, qmix).")
        from scripts.core.runners.run_cleanrl import run_cleanrl
        run_cleanrl(args, algorithm=args.algorithm)
        
    elif args.library == "rllib":
        if args.algorithm != "ppo":
            raise ValueError(f"Algorithm {args.algorithm} not supported by rllib runner (supported: ppo).")
        from scripts.core.runners.run_rllib import run_rllib
        run_rllib(args)
        
    else:
        raise ValueError(f"Unknown library: {args.library}")

if __name__ == "__main__":
    main()
