#!/usr/bin/env python3
"""Visualize and summarize training metrics from CSV logs.

Usage:
    uv run python -m scripts.visualize_logs
    uv run python -m scripts.visualize_logs logs/run_xxx
    uv run python -m scripts.visualize_logs --compare
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

THEME = {
    "fig": "#f6f3ed",
    "axes": "#fffdf8",
    "grid": "#d8d2c5",
    "text": "#352b1e",
    "muted": "#7e7060",
    "reward": "#c34a36",
    "food": "#3f7d20",
    "food_rem": "#d4a017",
    "coop": "#00798c",
    "eff": "#845ec2",
    "collisions": "#d65d0e",
    "failed": "#9c6644",
    "length": "#33658a",
    "agent_a": "#5d3fd3",
    "agent_b": "#00876c",
}


plt.rcParams.update(
    {
        "figure.facecolor": THEME["fig"],
        "axes.facecolor": THEME["axes"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["text"],
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "grid.color": THEME["grid"],
        "grid.alpha": 0.55,
        "xtick.color": THEME["muted"],
        "ytick.color": THEME["muted"],
        "text.color": THEME["text"],
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "legend.facecolor": "#fff8ec",
        "legend.edgecolor": THEME["grid"],
    }
)


def smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    """Centered moving average."""
    if len(values) <= 2:
        return values.astype(float)
    w = max(3, min(window, len(values)))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(values, kernel, mode="same")


def rolling_std(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) <= 2:
        return np.zeros_like(values, dtype=float)
    mean = smooth(values, window)
    mean_sq = smooth(values**2, window)
    var = np.maximum(0.0, mean_sq - mean**2)
    return np.sqrt(var)


def relative_progress(values: np.ndarray, baseline_count: int) -> np.ndarray:
    """Percent change from the early-training baseline."""
    if len(values) == 0:
        return np.array([], dtype=float)
    k = max(1, min(baseline_count, len(values)))
    baseline = float(np.mean(values[:k]))
    if abs(baseline) < 1e-9:
        return np.zeros_like(values, dtype=float)
    return ((values - baseline) / abs(baseline)) * 100.0


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    def as_int(key: str) -> np.ndarray:
        return np.array([int(r.get(key, 0) or 0) for r in rows], dtype=int)

    def as_float(key: str) -> np.ndarray:
        return np.array([float(r.get(key, 0.0) or 0.0) for r in rows], dtype=float)

    data: dict[str, np.ndarray] = {
        "episode": as_int("episode"),
        "timestep": as_int("timestep"),
        "wall_time": as_float("wall_time"),
        "reward": as_float("reward_total"),
        "length": as_int("length"),
        "foods_collected": as_int("foods_collected"),
        "coop_collections": as_int("cooperative_collections"),
        "solo_collections": as_int("solo_collections"),
        "failed_loads": as_int("failed_loads"),
        "food_remaining": as_int("food_remaining_end"),
        "collisions": as_int("collisions"),
        "actions_north": as_int("actions_north"),
        "actions_south": as_int("actions_south"),
        "actions_east": as_int("actions_east"),
        "actions_west": as_int("actions_west"),
        "actions_load": as_int("actions_load"),
        "actions_none": as_int("actions_none"),
    }

    # Per-agent rewards (if present)
    agent_rewards_list = [json.loads(r.get("agent_rewards", "{}")) for r in rows]
    if agent_rewards_list and isinstance(agent_rewards_list[0], dict):
        for name in sorted(agent_rewards_list[0].keys()):
            data[f"reward_{name}"] = np.array(
                [float(ar.get(name, 0.0)) for ar in agent_rewards_list], dtype=float
            )

    foods = np.maximum(data["foods_collected"], 1)
    lengths = np.maximum(data["length"], 1)
    data["coop_rate"] = data["coop_collections"] / foods
    data["efficiency"] = data["foods_collected"] / lengths

    total_actions = (
        data["actions_north"]
        + data["actions_south"]
        + data["actions_east"]
        + data["actions_west"]
        + data["actions_load"]
        + data["actions_none"]
    )
    safe_total = np.maximum(total_actions, 1)
    data["load_fraction"] = data["actions_load"] / safe_total
    data["move_fraction"] = (
        data["actions_north"] + data["actions_south"] + data["actions_east"] + data["actions_west"]
    ) / safe_total

    return data


def compute_summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    n = len(data["episode"])
    first_q_end = max(1, n // 4)
    first_q = slice(0, first_q_end)
    last_q_start = max(0, 3 * n // 4)
    last_q = slice(last_q_start, n)

    summary = {
        "episodes": float(n),
        "timesteps": float(data["timestep"][-1]) if n else 0.0,
        "reward_mean": float(np.mean(data["reward"])),
        "reward_first_q": float(np.mean(data["reward"][first_q])),
        "reward_last_q": float(np.mean(data["reward"][last_q])),
        "reward_last_q_std": float(np.std(data["reward"][last_q])),
        "reward_max": float(np.max(data["reward"])),
        "foods_mean": float(np.mean(data["foods_collected"])),
        "foods_first_q": float(np.mean(data["foods_collected"][first_q])),
        "foods_last_q": float(np.mean(data["foods_collected"][last_q])),
        "coop_mean": float(np.mean(data["coop_rate"])),
        "coop_last_q": float(np.mean(data["coop_rate"][last_q])),
        "eff_mean": float(np.mean(data["efficiency"])),
        "eff_first_q": float(np.mean(data["efficiency"][first_q])),
        "eff_last_q": float(np.mean(data["efficiency"][last_q])),
        "collisions_mean": float(np.mean(data["collisions"])),
        "failed_loads_mean": float(np.mean(data["failed_loads"])),
        "food_remaining_mean": float(np.mean(data["food_remaining"])),
        "length_mean": float(np.mean(data["length"])),
    }
    reward_base = max(abs(summary["reward_first_q"]), 1e-9)
    foods_base = max(abs(summary["foods_first_q"]), 1e-9)
    eff_base = max(abs(summary["eff_first_q"]), 1e-9)
    summary["reward_improvement_pct"] = (
        (summary["reward_last_q"] - summary["reward_first_q"]) / reward_base
    ) * 100.0
    summary["foods_improvement_pct"] = (
        (summary["foods_last_q"] - summary["foods_first_q"]) / foods_base
    ) * 100.0
    summary["eff_improvement_pct"] = (
        (summary["eff_last_q"] - summary["eff_first_q"]) / eff_base
    ) * 100.0
    return summary


def run_label(run_dir: Path, config: dict[str, Any]) -> str:
    preset = config.get("preset")
    algo = config.get("algorithm")
    if preset and algo:
        return f"{run_dir.name} [{algo}/{preset}]"
    if preset:
        return f"{run_dir.name} [{preset}]"
    return run_dir.name


def render_summary_text(
    ax: Any,
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, float],
) -> None:
    ax.axis("off")
    benchmark = config.get("benchmark", "n/a")
    preset = config.get("preset", "n/a")
    lr = config.get("learning_rate", "n/a")

    lines = [
        f"Run: {run_dir.name}",
        f"Benchmark: {benchmark}",
        f"Preset: {preset}    LR: {lr}",
        f"Episodes: {int(summary['episodes'])}    Steps: {int(summary['timesteps']):,}",
        "",
        f"Reward   {summary['reward_first_q']:.2f} -> {summary['reward_last_q']:.2f} ({summary['reward_improvement_pct']:+.1f}%)",
        f"Foods    {summary['foods_first_q']:.2f} -> {summary['foods_last_q']:.2f} ({summary['foods_improvement_pct']:+.1f}%)",
        f"Eff      {summary['eff_first_q']:.4f} -> {summary['eff_last_q']:.4f} ({summary['eff_improvement_pct']:+.1f}%)",
        f"Coop     {summary['coop_mean']:.1%} mean, {summary['coop_last_q']:.1%} late",
        f"Friction collisions={summary['collisions_mean']:.2f}, failed={summary['failed_loads_mean']:.2f}",
    ]

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "#fff4df", "edgecolor": THEME["grid"], "boxstyle": "round,pad=0.7"},
    )


def plot_single_run(log_dir: Path, window: int = 25, save: bool = True, show: bool = True) -> None:
    csv_path = log_dir / "metrics.csv"
    config_path = log_dir / "config.json"

    if not csv_path.exists():
        print(f"No metrics.csv in {log_dir}")
        return

    data = load_csv(csv_path)
    if not data:
        print("No episode data recorded.")
        return

    config = load_config(config_path)
    summary = compute_summary(data)
    ts = data["timestep"]
    n = len(ts)

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.25)

    fig.suptitle(f"Sustainable Foraging Dashboard - {run_label(log_dir, config)}", y=0.985)

    # 1) Reward trend with confidence band
    ax = fig.add_subplot(gs[0, 0])
    reward_sm = smooth(data["reward"], window)
    reward_sd = rolling_std(data["reward"], window)
    first_q_end = max(1, n // 4)
    last_q_start = max(0, 3 * n // 4)
    ax.plot(ts, data["reward"], color=THEME["reward"], alpha=0.18, linewidth=1)
    ax.plot(ts, reward_sm, color=THEME["reward"], linewidth=2.3, label="Smoothed reward")
    ax.fill_between(
        ts, reward_sm - reward_sd, reward_sm + reward_sd, color=THEME["reward"], alpha=0.14
    )
    if n >= 4:
        ax.axvspan(ts[0], ts[first_q_end - 1], color="#f0e7d9", alpha=0.55, label="First 25%")
        ax.axvspan(ts[last_q_start], ts[-1], color="#e9f6e7", alpha=0.45, label="Last 25%")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend()

    # 2) Foods vs remaining
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(
        ts,
        smooth(data["foods_collected"], window),
        color=THEME["food"],
        linewidth=2.2,
        label="Foods collected",
    )
    ax.plot(
        ts,
        smooth(data["food_remaining"], window),
        color=THEME["food_rem"],
        linewidth=2.2,
        label="Food remaining",
    )
    env_cfg = config.get("environment", {}) if isinstance(config, dict) else {}
    max_food = env_cfg.get("max_num_food")
    if isinstance(max_food, (int, float)):
        ax.axhline(
            y=max_food,
            linestyle="--",
            color=THEME["muted"],
            alpha=0.6,
            label=f"Max food={max_food}",
        )
    ax.set_title("Food Dynamics")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend()

    # 3) Cooperation + efficiency
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(
        ts,
        smooth(data["coop_rate"], window),
        color=THEME["coop"],
        linewidth=2.2,
        label="Cooperation rate",
    )
    ax.plot(
        ts,
        smooth(data["efficiency"], window),
        color=THEME["eff"],
        linewidth=2.2,
        label="Efficiency",
    )
    ax.set_ylim(bottom=-0.02)
    ax.set_title("Teamwork and Efficiency")
    ax.set_xlabel("Timesteps")
    ax.grid(True)
    ax.legend()

    # 4) Episode length
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts, data["length"], color=THEME["length"], alpha=0.2)
    ax.plot(ts, smooth(data["length"], window), color=THEME["length"], linewidth=2.2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Steps")
    ax.grid(True)

    # 5) Friction metrics
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(
        ts,
        smooth(data["collisions"], window),
        color=THEME["collisions"],
        linewidth=2.2,
        label="Collisions",
    )
    ax.plot(
        ts,
        smooth(data["failed_loads"], window),
        color=THEME["failed"],
        linewidth=2.2,
        label="Failed loads",
    )
    ax.set_title("Coordination Friction")
    ax.set_xlabel("Timesteps")
    ax.grid(True)
    ax.legend()

    # 6) Action distribution
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(0, n, num=min(120, max(12, n // 4)), dtype=int)
    bins = np.unique(np.clip(bins, 0, n))
    if len(bins) < 2:
        bins = np.array([0, n], dtype=int)
    action_names = ["north", "south", "east", "west", "load", "none"]
    colors = ["#7aa6c2", "#8bc3d8", "#7fbf7f", "#a5d296", "#d28d49", "#b7b2aa"]
    stacked: list[np.ndarray] = []
    xvals = ts[bins[:-1]] if len(ts) else np.array([])
    for name in action_names:
        key = f"actions_{name}"
        values = []
        for i in range(len(bins) - 1):
            b0, b1 = bins[i], bins[i + 1]
            if b1 <= b0:
                values.append(0.0)
            else:
                values.append(float(np.mean(data[key][b0:b1])))
        stacked.append(np.array(values, dtype=float))
    mat = np.vstack(stacked)
    denom = np.maximum(mat.sum(axis=0, keepdims=True), 1.0)
    mat = mat / denom
    ax.stackplot(xvals, mat, labels=[n.upper() for n in action_names], colors=colors, alpha=0.9)
    ax.set_title("Action Mix (windowed fractions)")
    ax.set_xlabel("Timesteps")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper right", ncol=2, fontsize=7)

    # 7) Per-agent rewards
    ax = fig.add_subplot(gs[2, 0])
    agent_keys = sorted([k for k in data if k.startswith("reward_agent_")])
    palette = [THEME["agent_a"], THEME["agent_b"], "#b56576", "#2a9d8f", "#264653"]
    if agent_keys:
        for idx, key in enumerate(agent_keys):
            ax.plot(
                ts,
                smooth(data[key], window),
                linewidth=2.1,
                label=key.replace("reward_", ""),
                color=palette[idx % len(palette)],
            )
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "No per-agent rewards in CSV", ha="center", va="center", color=THEME["muted"]
        )
    ax.set_title("Per-Agent Reward")
    ax.set_xlabel("Timesteps")
    ax.grid(True)

    # 8) Improvement curves over training
    ax = fig.add_subplot(gs[2, 1])
    baseline_count = max(5, n // 10)
    reward_prog = smooth(relative_progress(data["reward"], baseline_count), window)
    foods_prog = smooth(relative_progress(data["foods_collected"], baseline_count), window)
    eff_prog = smooth(relative_progress(data["efficiency"], baseline_count), window)
    ax.axhline(0.0, color=THEME["muted"], linestyle="--", linewidth=1.2)
    ax.plot(ts, reward_prog, color=THEME["reward"], linewidth=2.2, label="Reward progress")
    ax.plot(ts, foods_prog, color=THEME["food"], linewidth=2.2, label="Foods progress")
    ax.plot(ts, eff_prog, color=THEME["eff"], linewidth=2.2, label="Efficiency progress")
    ax.set_title("Improvement vs Early Training (%)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Change from first 10%")
    ax.grid(True)
    ax.legend()

    # 9) Summary card
    ax = fig.add_subplot(gs[2, 2])
    render_summary_text(ax, log_dir, config, summary)

    if save:
        out = log_dir / "training_dashboard.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved: {out}")

    print("\n" + "-" * 66)
    print(f"Run summary: {run_label(log_dir, config)}")
    print("-" * 66)
    print(f"Episodes: {int(summary['episodes'])}")
    print(f"Timesteps: {int(summary['timesteps']):,}")
    print(
        f"Reward first25 -> last25: {summary['reward_first_q']:.3f} -> "
        f"{summary['reward_last_q']:.3f} ({summary['reward_improvement_pct']:+.1f}%)"
    )
    print(
        f"Foods first25 -> last25: {summary['foods_first_q']:.3f} -> "
        f"{summary['foods_last_q']:.3f} ({summary['foods_improvement_pct']:+.1f}%)"
    )
    print(
        f"Efficiency first25 -> last25: {summary['eff_first_q']:.4f} -> "
        f"{summary['eff_last_q']:.4f} ({summary['eff_improvement_pct']:+.1f}%)"
    )
    print(
        f"Reward mean / last25 / max: {summary['reward_mean']:.3f} / "
        f"{summary['reward_last_q']:.3f} / {summary['reward_max']:.3f}"
    )
    print(f"Foods mean / last25: {summary['foods_mean']:.3f} / {summary['foods_last_q']:.3f}")
    print(f"Coop mean / last25: {summary['coop_mean']:.1%} / {summary['coop_last_q']:.1%}")
    print(f"Efficiency mean / last25: {summary['eff_mean']:.4f} / {summary['eff_last_q']:.4f}")
    print(
        f"Collisions mean: {summary['collisions_mean']:.2f}, "
        f"Failed loads mean: {summary['failed_loads_mean']:.2f}"
    )
    print("-" * 66)

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_comparison_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fields = [
        "run",
        "benchmark",
        "algorithm",
        "preset",
        "episodes",
        "timesteps",
        "reward_mean",
        "reward_last_q",
        "reward_last_q_std",
        "foods_mean",
        "foods_last_q",
        "coop_mean",
        "coop_last_q",
        "eff_mean",
        "eff_last_q",
        "collisions_mean",
        "failed_loads_mean",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved: {out_path}")


def plot_compare(
    logs_root: Path,
    window: int = 25,
    save: bool = True,
    show: bool = True,
    limit: int | None = None,
) -> None:
    runs = sorted([r for r in logs_root.glob("run_*") if (r / "metrics.csv").exists()])
    if not runs:
        print(f"No completed runs in {logs_root}/")
        return

    run_rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []

    for run_dir in runs:
        data = load_csv(run_dir / "metrics.csv")
        if not data:
            continue
        config = load_config(run_dir / "config.json")
        summary = compute_summary(data)
        row = {
            "run": run_dir.name,
            "benchmark": config.get("benchmark", ""),
            "algorithm": config.get("algorithm", ""),
            "preset": config.get("preset", ""),
            **summary,
        }
        run_rows.append(row)
        series.append({"run_dir": run_dir, "config": config, "data": data, "summary": summary})

    if not series:
        print("No readable run data.")
        return

    series.sort(key=lambda x: x["summary"]["reward_last_q"], reverse=True)
    if limit is not None and limit > 0:
        series = series[:limit]

    colors = [
        "#C34A36",
        "#2D8F85",
        "#845EC2",
        "#FF8066",
        "#4D8076",
        "#6A4C93",
        "#3D5A80",
        "#9C6644",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Sustainable Foraging: Run Comparison", y=0.98)

    # A) Reward curves
    ax = axes[0, 0]
    for i, s in enumerate(series):
        ts = s["data"]["timestep"]
        label = run_label(s["run_dir"], s["config"])
        ax.plot(
            ts,
            smooth(s["data"]["reward"], window),
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=label,
        )
    ax.set_title("Reward (smoothed)")
    ax.set_xlabel("Timesteps")
    ax.grid(True)

    # B) Efficiency curves
    ax = axes[0, 1]
    for i, s in enumerate(series):
        ts = s["data"]["timestep"]
        label = run_label(s["run_dir"], s["config"])
        ax.plot(
            ts,
            smooth(s["data"]["efficiency"], window),
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=label,
        )
    ax.set_title("Collection Efficiency (smoothed)")
    ax.set_xlabel("Timesteps")
    ax.grid(True)

    # C) Cooperation curves
    ax = axes[1, 0]
    for i, s in enumerate(series):
        ts = s["data"]["timestep"]
        label = run_label(s["run_dir"], s["config"])
        ax.plot(
            ts,
            smooth(s["data"]["coop_rate"], window),
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=label,
        )
    ax.set_title("Cooperation Rate (smoothed)")
    ax.set_xlabel("Timesteps")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)

    # D) Ranking bar chart
    ax = axes[1, 1]
    labels = [run_label(s["run_dir"], s["config"]) for s in series]
    scores = [s["summary"]["reward_last_q"] for s in series]
    errs = [s["summary"]["reward_last_q_std"] for s in series]
    y = np.arange(len(labels))
    ax.barh(
        y, scores, xerr=errs, color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.9
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Ranking by Last 25% Reward")
    ax.set_xlabel("Reward (mean +/- std over last quarter)")
    ax.grid(True, axis="x")

    for a in axes.flat:
        if a in (axes[0, 0], axes[0, 1], axes[1, 0]):
            a.legend(fontsize=7, loc="best")

    plt.tight_layout()

    if save:
        out_png = logs_root / "comparison_dashboard.png"
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        print(f"Saved: {out_png}")

    # Print leaderboard
    print("\n" + "=" * 92)
    print("Leaderboard (sorted by last 25% reward)")
    print("=" * 92)
    print(
        f"{'Run':28} {'Preset':8} {'Algo':8} {'R_last25':>10} {'Foods':>8} {'Coop':>8} {'Eff':>10}"
    )
    print("-" * 92)
    for s in series:
        cfg = s["config"]
        summ = s["summary"]
        print(
            f"{s['run_dir'].name:28} "
            f"{cfg.get('preset', '')!s:8} "
            f"{cfg.get('algorithm', '')!s:8} "
            f"{summ['reward_last_q']:10.3f} "
            f"{summ['foods_last_q']:8.3f} "
            f"{summ['coop_last_q']:8.1%} "
            f"{summ['eff_last_q']:10.4f}"
        )
    print("=" * 92)

    export_comparison_csv(run_rows, logs_root / "comparison_summary.csv")

    if show:
        plt.show()
    else:
        plt.close(fig)


def find_latest_run(logs_root: Path) -> Path | None:
    runs = sorted([r for r in logs_root.glob("run_*") if (r / "metrics.csv").exists()])
    return runs[-1] if runs else None


def backend_is_interactive() -> bool:
    backend = plt.get_backend().lower()
    non_interactive_tokens = ("agg", "pdf", "ps", "svg", "cairo")
    return not any(token in backend for token in non_interactive_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LB-Foraging training")
    parser.add_argument("log_dir", nargs="?", default=None, help="Run directory")
    parser.add_argument("--compare", action="store_true", help="Compare completed runs")
    parser.add_argument("--window", type=int, default=25, help="Smoothing window (default: 25)")
    parser.add_argument(
        "--top", type=int, default=0, help="In compare mode, show top N runs (0 = all)"
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG outputs")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open interactive plot window"
    )
    args = parser.parse_args()

    logs_root = Path("logs")
    save = not args.no_save
    show = not args.no_show
    if show and not backend_is_interactive():
        print("Non-interactive Matplotlib backend detected; skipping interactive window.")
        show = False
    top = args.top if args.top > 0 else None

    if args.compare:
        plot_compare(logs_root, window=args.window, save=save, show=show, limit=top)
        return

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_run(logs_root)
        if log_dir is None:
            print("No runs found. Train first: uv run python -m scripts.train_sb3")
            sys.exit(1)
        print(f"Using latest run: {log_dir}")

    plot_single_run(log_dir, window=args.window, save=save, show=show)


if __name__ == "__main__":
    main()
