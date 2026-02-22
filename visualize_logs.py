#!/usr/bin/env python3
"""Visualize training metrics from CSV logs.

Usage:
    uv run python visualize_logs.py                    # latest run
    uv run python visualize_logs.py logs/run_xxx       # specific run
    uv run python visualize_logs.py --compare          # compare all runs
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Styling
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#eee",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.4,
    "lines.linewidth": 1.2,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#444",
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.size": 9,
})

COLORS = {
    "primary": "#e94560",
    "secondary": "#53d8fb",
    "accent": "#f5a623",
    "green": "#7ed957",
    "purple": "#bd93f9",
    "pink": "#ff79c6",
    "orange": "#ffb86c",
    "cyan": "#8be9fd",
}


# Data loading
def load_csv(csv_path: str) -> dict:
    """Load metrics CSV into a dict of numpy arrays."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    data = {
        "episode": np.array([int(r["episode"]) for r in rows]),
        "timestep": np.array([int(r["timestep"]) for r in rows]),
        "wall_time": np.array([float(r["wall_time"]) for r in rows]),
        "reward": np.array([float(r["reward_total"]) for r in rows]),
        "length": np.array([int(r["length"]) for r in rows]),
        "foods_collected": np.array([int(r["foods_collected"]) for r in rows]),
        "coop_collections": np.array([int(r["cooperative_collections"]) for r in rows]),
        "solo_collections": np.array([int(r["solo_collections"]) for r in rows]),
        "failed_loads": np.array([int(r["failed_loads"]) for r in rows]),
        "food_remaining": np.array([int(r["food_remaining_end"]) for r in rows]),
        "collisions": np.array([int(r["collisions"]) for r in rows]),
        "actions_north": np.array([int(r["actions_north"]) for r in rows]),
        "actions_south": np.array([int(r["actions_south"]) for r in rows]),
        "actions_east": np.array([int(r["actions_east"]) for r in rows]),
        "actions_west": np.array([int(r["actions_west"]) for r in rows]),
        "actions_load": np.array([int(r["actions_load"]) for r in rows]),
        "actions_none": np.array([int(r["actions_none"]) for r in rows]),
    }

    # Per-agent rewards
    agent_rewards_list = [json.loads(r["agent_rewards"]) for r in rows]
    if agent_rewards_list and agent_rewards_list[0]:
        agent_names = sorted(agent_rewards_list[0].keys())
        for name in agent_names:
            data[f"reward_{name}"] = np.array([ar.get(name, 0.0) for ar in agent_rewards_list])

    # Derived
    data["coop_rate"] = data["coop_collections"] / np.maximum(data["foods_collected"], 1)
    data["efficiency"] = data["foods_collected"] / np.maximum(data["length"], 1)
    total_actions = (
        data["actions_north"] + data["actions_south"] +
        data["actions_east"] + data["actions_west"] +
        data["actions_load"] + data["actions_none"]
    )
    data["load_fraction"] = data["actions_load"] / np.maximum(total_actions, 1)
    data["move_fraction"] = (
        data["actions_north"] + data["actions_south"] +
        data["actions_east"] + data["actions_west"]
    ) / np.maximum(total_actions, 1)

    return data


def smooth(values, window=20):
    """Exponential moving average."""
    if len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


# Plotting: single run (full dashboard)
def plot_single_run(log_dir: Path, save: bool = True):
    csv_path = log_dir / "metrics.csv"
    config_path = log_dir / "config.json"
    if not csv_path.exists():
        print(f"No metrics.csv in {log_dir}")
        return

    data = load_csv(str(csv_path))
    if not data:
        print("No episode data recorded.")
        return

    n = len(data["episode"])
    ts = data["timestep"]
    print(f"Loaded {n} episodes from {csv_path}")

    # Load config if available
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # --- Create figure with GridSpec ---
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    title = f"Training Dashboard — {log_dir.name}"
    if config:
        title += f"  ({config.get('total_timesteps', '?'):,} steps, lr={config.get('learning_rate', '?')})"
    fig.suptitle(title, y=0.98)

    # ── 1. Episode Reward ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, data["reward"], alpha=0.15, color=COLORS["primary"])
    ax.plot(ts, smooth(data["reward"]), color=COLORS["primary"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True)

    # ── 2. Episode Length ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ts, data["length"], alpha=0.15, color=COLORS["secondary"])
    ax.plot(ts, smooth(data["length"]), color=COLORS["secondary"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.legend()
    ax.grid(True)

    # ── 3. Foods Collected Per Episode ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ts, data["foods_collected"], alpha=0.15, color=COLORS["green"])
    ax.plot(ts, smooth(data["foods_collected"]), color=COLORS["green"], linewidth=2, label="Smoothed")
    food_total = config.get("environment", {}).get("max_num_food", "?")
    ax.axhline(y=food_total if isinstance(food_total, (int, float)) else 0,
               color=COLORS["accent"], linestyle="--", alpha=0.5, label=f"Max ({food_total})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Foods")
    ax.set_title("Foods Collected")
    ax.legend()
    ax.grid(True)

    # ── 4. Cooperation Rate ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts, data["coop_rate"], alpha=0.15, color=COLORS["purple"])
    ax.plot(ts, smooth(data["coop_rate"]), color=COLORS["purple"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate")
    ax.set_title("Cooperation Rate (joint / total collections)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True)

    # ── 5. Collisions Per Episode ──
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts, data["collisions"], alpha=0.15, color=COLORS["orange"])
    ax.plot(ts, smooth(data["collisions"]), color=COLORS["orange"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Count")
    ax.set_title("Collisions Per Episode")
    ax.legend()
    ax.grid(True)

    # ── 6. Food Remaining at Episode End ──
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ts, data["food_remaining"], alpha=0.15, color=COLORS["accent"])
    ax.plot(ts, smooth(data["food_remaining"]), color=COLORS["accent"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Remaining")
    ax.set_title("Food Remaining at Episode End")
    ax.legend()
    ax.grid(True)

    # ── 7. Action Distribution (stacked area) ──
    ax = fig.add_subplot(gs[2, 0])
    window = max(1, n // 100)
    bins = np.arange(0, n, window)
    action_names = ["NORTH", "SOUTH", "EAST", "WEST", "LOAD", "NONE"]
    action_keys = [f"actions_{a.lower()}" for a in action_names]
    action_colors = [COLORS["secondary"], COLORS["cyan"], COLORS["green"],
                     COLORS["accent"], COLORS["primary"], "#666"]

    stacked = np.zeros((len(bins), len(action_names)))
    for j, key in enumerate(action_keys):
        for k, b in enumerate(bins):
            end = min(b + window, n)
            stacked[k, j] = data[key][b:end].mean()

    # Normalize to fractions
    row_sums = stacked.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    stacked_norm = stacked / row_sums

    bin_ts = ts[bins] if len(bins) <= len(ts) else bins
    ax.stackplot(bin_ts, stacked_norm.T, labels=action_names, colors=action_colors, alpha=0.8)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Fraction")
    ax.set_title("Action Distribution")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(True)

    # ── 8. Per-Agent Rewards ──
    ax = fig.add_subplot(gs[2, 1])
    agent_keys = [k for k in data if k.startswith("reward_agent_")]
    agent_colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["green"],
                         COLORS["accent"], COLORS["purple"]]
    for j, key in enumerate(agent_keys):
        label = key.replace("reward_", "")
        c = agent_colors_list[j % len(agent_colors_list)]
        ax.plot(ts, smooth(data[key]), color=c, linewidth=2, label=label)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Per-Agent Reward (smoothed)")
    ax.legend()
    ax.grid(True)

    # ── 9. Collection Efficiency ──
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(ts, data["efficiency"], alpha=0.15, color=COLORS["pink"])
    ax.plot(ts, smooth(data["efficiency"]), color=COLORS["pink"], linewidth=2, label="Smoothed")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Foods / Step")
    ax.set_title("Collection Efficiency")
    ax.legend()
    ax.grid(True)

    if save:
        out = log_dir / "training_dashboard.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    # --- Summary ---
    last_q = slice(3 * n // 4, n)
    print(f"\n{'━' * 55}")
    print(f"  Summary — {log_dir.name}")
    print(f"{'━' * 55}")
    print(f"  Episodes          : {n}")
    print(f"  Timesteps         : {ts[-1]:,}")
    print(f"  Mean reward       : {data['reward'].mean():.4f}")
    print(f"  Last 25% reward   : {data['reward'][last_q].mean():.4f}")
    print(f"  Max reward        : {data['reward'].max():.4f}")
    print(f"  Mean length       : {data['length'].mean():.1f}")
    print(f"  Mean foods/ep     : {data['foods_collected'].mean():.2f}")
    print(f"  Cooperation rate  : {data['coop_rate'].mean():.2%}")
    print(f"  Mean collisions   : {data['collisions'].mean():.1f}")
    print(f"  Collection eff.   : {data['efficiency'].mean():.4f} foods/step")
    print(f"{'━' * 55}")

    plt.show()


# Plotting: compare runs
def plot_compare(logs_root: Path, save: bool = True):
    runs = sorted(logs_root.glob("run_*"))
    runs = [r for r in runs if (r / "metrics.csv").exists()]

    if not runs:
        print(f"No completed runs in {logs_root}/")
        return

    colors_list = list(COLORS.values())

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Run Comparison ({len(runs)} runs)", y=0.98)

    metrics = [
        ("reward", "Episode Reward"),
        ("length", "Episode Length"),
        ("foods_collected", "Foods Collected"),
        ("coop_rate", "Cooperation Rate"),
        ("collisions", "Collisions"),
        ("efficiency", "Collection Efficiency"),
    ]

    for run_idx, run_dir in enumerate(runs):
        data = load_csv(str(run_dir / "metrics.csv"))
        if not data:
            continue

        color = colors_list[run_idx % len(colors_list)]
        label = run_dir.name
        ts = data["timestep"]

        for ax_idx, (key, title) in enumerate(metrics):
            ax = axes[ax_idx // 3, ax_idx % 3]
            ax.plot(ts, smooth(data[key]), color=color, linewidth=2, label=label, alpha=0.9)
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.grid(True)

    for ax in axes.flat:
        ax.legend(fontsize=7)

    plt.tight_layout()

    if save:
        out = logs_root / "comparison.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()


# CLI
def find_latest_run(logs_root: Path) -> Path | None:
    runs = sorted(logs_root.glob("run_*"))
    runs = [r for r in runs if (r / "metrics.csv").exists()]
    return runs[-1] if runs else None


def main():
    parser = argparse.ArgumentParser(description="Visualize LB-Foraging training")
    parser.add_argument("log_dir", nargs="?", default=None, help="Run directory")
    parser.add_argument("--compare", action="store_true", help="Compare all runs")
    args = parser.parse_args()

    logs_root = Path("logs")

    if args.compare:
        plot_compare(logs_root)
        return

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_run(logs_root)
        if not log_dir:
            print("No runs found. Train first: uv run python train_sb3.py")
            sys.exit(1)
        print(f"Using latest run: {log_dir}")

    plot_single_run(log_dir)


if __name__ == "__main__":
    main()
