#!/usr/bin/env python3
"""Compare models (Library/Algorithm) on a single dashboard.

Each run appears as its own entry, labeled by its library and algorithm
(e.g. SB3/PPO, RLlib/DQN, CleanRL/PPO).

Usage:
    # Compare all runs in logs/
    uv run python -m scripts.compare_algorithms

    # Filter by preset
    uv run python -m scripts.compare_algorithms --preset fair

    # Compare specific runs
    uv run python -m scripts.compare_algorithms logs/run_a logs/run_b

    # Top N only
    uv run python -m scripts.compare_algorithms --top 5
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

# ── Theme ────────────────────────────────────────────────────────────────────

PALETTE = [
    "#C34A36",
    "#2D8F85",
    "#845EC2",
    "#D4A017",
    "#3D5A80",
    "#9C6644",
    "#00876C",
    "#B56576",
    "#264653",
    "#6A4C93",
]

BG_FIG = "#f6f3ed"
BG_AXES = "#fffdf8"
GRID_CLR = "#d8d2c5"
TEXT_CLR = "#352b1e"
MUTED = "#7e7060"

plt.rcParams.update(
    {
        "figure.facecolor": BG_FIG,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": GRID_CLR,
        "axes.labelcolor": TEXT_CLR,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "grid.color": GRID_CLR,
        "grid.alpha": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT_CLR,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "legend.facecolor": "#fff8ec",
        "legend.edgecolor": GRID_CLR,
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def smooth(v: np.ndarray, window: int = 25) -> np.ndarray:
    if len(v) <= 2:
        return v.astype(float)
    w = max(3, min(window, len(v)))
    w += 1 - w % 2  # ensure odd
    # Pad at data edges to prevent np.convolve from padding with zeros.
    # We use "reflect" instead of "edge" so that if the very last episode is
    # an outlier (e.g. 0.0), we don't duplicate it 250 times and tank the graph!
    pad_w = w // 2
    padded_v = np.pad(v, (pad_w, pad_w), mode="reflect")
    return np.convolve(padded_v, np.ones(w) / w, mode="valid")


def rolling_std(v: np.ndarray, window: int = 25) -> np.ndarray:
    mu = smooth(v, window)
    return np.sqrt(np.maximum(0.0, smooth(v**2, window) - mu**2))


def _label(config: dict[str, Any], run_dir: Path) -> str:
    lib = config.get("library", "")
    algo = config.get("algorithm", "")
    if lib and algo:
        return f"{lib}/{algo}"
    if algo:
        return algo
    return run_dir.name


def load_run(run_dir: Path, max_steps: int = 0) -> dict[str, Any] | None:
    csv_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "config.json"
    if not csv_path.exists():
        return None

    config: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)

    rows: list[dict[str, str]] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts = int(row.get("timestep", 0) or 0)
            if max_steps > 0 and ts > max_steps:
                continue
            rows.append(row)
    if not rows:
        return None

    def col_f(k: str) -> np.ndarray:
        return np.array([float(r.get(k, 0) or 0) for r in rows])

    def col_i(k: str) -> np.ndarray:
        return np.array([int(r.get(k, 0) or 0) for r in rows])

    ts = col_i("timestep")
    rew = col_f("reward_total")
    food = col_i("foods_collected")
    leng = col_i("length")
    rem = col_i("food_remaining_end")

    sl = np.maximum(leng, 1)
    n = len(ts)
    q3 = max(0, 3 * n // 4)

    return {
        "run_dir": run_dir,
        "config": config,
        "label": _label(config, run_dir),
        "preset": config.get("preset", "?"),
        "ts": ts,
        "reward": rew,
        "foods": food.astype(float),
        "remaining": rem.astype(float),
        "efficiency": food / sl,
        # Scalars (last 25 %)
        "reward_lq": float(np.mean(rew[q3:])),
        "reward_lq_sd": float(np.std(rew[q3:])),
        "foods_lq": float(np.mean(food[q3:])),
        "eff_lq": float(np.mean(food[q3:] / sl[q3:])),
        "rem_lq": float(np.mean(rem[q3:])),
    }


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot(runs: list[dict[str, Any]], out: Path, window: int, show: bool, lines_only: bool = False) -> None:
    runs.sort(key=lambda r: r["reward_lq"], reverse=True)
    N = len(runs)
    clr = [PALETTE[i % len(PALETTE)] for i in range(N)]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    presets = sorted(set(r["preset"] for r in runs))
    title = "Model Comparison"
    if len(presets) == 1:
        title += f"  —  {presets[0]} preset"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)

    # 1 ─ Reward curves
    ax = fig.add_subplot(gs[0, 0:2])
    for i, r in enumerate(runs):
        mu = smooth(r["reward"], window)
        if not lines_only:
            sd = rolling_std(r["reward"], window)
            ax.fill_between(r["ts"], mu - sd, mu + sd, color=clr[i], alpha=0.10)
        ax.plot(r["ts"], mu, color=clr[i], lw=2.2, label=r["label"])
    ax.set_title("Episode Reward")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best", ncol=max(1, N // 5))

    # 2 ─ Ranking bar
    ax = fig.add_subplot(gs[0, 2])
    labels = [r["label"] for r in runs]
    scores = [r["reward_lq"] for r in runs]
    errs = [r["reward_lq_sd"] for r in runs]
    y = np.arange(N)
    bars = ax.barh(y, scores, xerr=errs, color=clr, alpha=0.9, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title("Final Reward (last 25%)")
    ax.set_xlabel("Mean ± std")
    ax.grid(True, axis="x")
    for b, s in zip(bars, scores):
        ax.text(
            b.get_width() + max(scores) * 0.02,
            b.get_y() + b.get_height() / 2,
            f"{s:.2f}",
            va="center",
            fontsize=9,
            color=TEXT_CLR,
        )

    # 3 ─ Efficiency
    ax = fig.add_subplot(gs[1, 0])
    for i, r in enumerate(runs):
        mu = smooth(r["efficiency"], window)
        if not lines_only:
            sd = rolling_std(r["efficiency"], window)
            ax.fill_between(r["ts"], mu - sd, mu + sd, color=clr[i], alpha=0.10)
        ax.plot(r["ts"], mu, color=clr[i], lw=2, label=r["label"])
    ax.set_title("Collection Efficiency")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Foods / Step")
    ax.grid(True)
    ax.legend(fontsize=7)

    # 4 ─ Foods collected
    ax = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(runs):
        mu = smooth(r["foods"], window)
        if not lines_only:
            sd = rolling_std(r["foods"], window)
            ax.fill_between(r["ts"], mu - sd, mu + sd, color=clr[i], alpha=0.10)
        ax.plot(r["ts"], mu, color=clr[i], lw=2, label=r["label"])
    ax.set_title("Foods Collected")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Count")
    ax.grid(True)
    ax.legend(fontsize=7)

    # 5 ─ Food remaining (sustainability)
    ax = fig.add_subplot(gs[1, 2])
    for i, r in enumerate(runs):
        mu = smooth(r["remaining"], window)
        if not lines_only:
            sd = rolling_std(r["remaining"], window)
            ax.fill_between(r["ts"], mu - sd, mu + sd, color=clr[i], alpha=0.10)
        ax.plot(r["ts"], mu, color=clr[i], lw=2, label=r["label"])
    env_cfg = runs[0]["config"].get("environment", {})
    K = env_cfg.get("max_num_food")
    if isinstance(K, (int, float)):
        ax.axhline(K, ls="--", color=MUTED, alpha=0.6, label=f"K={K}")
    ax.axhline(0, ls=":", color="#c34a36", alpha=0.5, label="Collapse")
    ax.set_title("Food Remaining (Sustainability)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Food at episode end")
    ax.grid(True)
    ax.legend(fontsize=7)

    # 6 ─ Scoreboard
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    hdrs = ["Model", "Preset", "Reward ↓", "Foods", "Efficiency", "Food Left"]
    rows_t = []
    row_clr = []
    for i, r in enumerate(runs):
        rows_t.append(
            [
                r["label"],
                r["preset"],
                f"{r['reward_lq']:.2f} ± {r['reward_lq_sd']:.2f}",
                f"{r['foods_lq']:.1f}",
                f"{r['eff_lq']:.4f}",
                f"{r['rem_lq']:.1f}",
            ]
        )
        row_clr.append([clr[i] + "18"] * len(hdrs))

    tbl = ax.table(
        cellText=rows_t,
        colLabels=hdrs,
        cellColours=row_clr,
        colColours=["#f0e7d9"] * len(hdrs),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for k, c in tbl.get_celld().items():
        if k[0] == 0:
            c.set_text_props(fontweight="bold")
    ax.set_title("Leaderboard", fontsize=13, fontweight="bold", pad=20)

    # Save
    png = out / "algorithm_comparison.png"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"Saved: {png}")

    # Console
    print()
    W = 94
    print("=" * W)
    print("  MODEL COMPARISON")
    print("=" * W)
    print(
        f"  {'#':>2}  {'Model':<18} {'Preset':<8} {'Reward':>14}  {'Foods':>6}  "
        f"{'Eff':>8}  {'Food Left':>9}"
    )
    print("-" * W)
    for i, r in enumerate(runs):
        print(
            f"  {i + 1:>2}  {r['label']:<18} {r['preset']:<8} "
            f"{r['reward_lq']:>7.2f}±{r['reward_lq_sd']:<5.2f}  "
            f"{r['foods_lq']:>6.1f}  {r['eff_lq']:>8.4f}  "
            f"{r['rem_lq']:>9.1f}"
        )
    print("=" * W)

    # CSV
    csv_out = out / "algorithm_comparison.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "model",
                "preset",
                "reward_mean",
                "reward_std",
                "foods",
                "efficiency",
                "food_remaining",
                "run_dir",
            ]
        )
        for i, r in enumerate(runs):
            w.writerow(
                [
                    i + 1,
                    r["label"],
                    r["preset"],
                    f"{r['reward_lq']:.4f}",
                    f"{r['reward_lq_sd']:.4f}",
                    f"{r['foods_lq']:.2f}",
                    f"{r['eff_lq']:.4f}",
                    f"{r['rem_lq']:.2f}",
                    r["run_dir"],
                ]
            )
    print(f"Saved: {csv_out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare models on a single dashboard")
    ap.add_argument(
        "run_dirs", nargs="*", default=[], help="Specific run dirs (default: all in logs/)"
    )
    ap.add_argument("--preset", type=str, default=None, help="Filter by preset")
    ap.add_argument("--top", type=int, default=0, help="Top N models (0=all)")
    ap.add_argument("--window", type=int, default=25, help="Smoothing window")
    ap.add_argument("--max-steps", type=int, default=0, help="Clip all graphs to this maximum timestep (e.g. 200000)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--lines-only", action="store_true", help="Plot only solid lines, no std dev bands")
    args = ap.parse_args()

    if args.run_dirs:
        dirs = [Path(d) for d in args.run_dirs]
    else:
        dirs = sorted(d for d in Path("logs").glob("run_*") if (d / "metrics.csv").exists())

    if not dirs:
        print("No runs found. Train first:\n  uv run python -m scripts.train_sb3 --preset fair")
        sys.exit(1)

    runs = [r for d in dirs if (r := load_run(d, args.max_steps)) is not None]
    if not runs:
        print("No valid run data.")
        sys.exit(1)

    if args.preset:
        runs = [r for r in runs if r["preset"] == args.preset]
        if not runs:
            print(f"No runs with preset '{args.preset}'")
            sys.exit(1)

    runs.sort(key=lambda r: r["reward_lq"], reverse=True)
    if args.top > 0:
        runs = runs[: args.top]

    print(f"Comparing {len(runs)} model(s)...")

    out = Path("logs")
    out.mkdir(exist_ok=True)
    backend = plt.get_backend().lower()
    show = not args.no_show and not any(t in backend for t in ("agg", "pdf", "svg", "cairo"))
    if not show and not args.no_show:
        print("Non-interactive backend; saving to file only.")

    plot(runs, out, window=args.window, show=show, lines_only=args.lines_only)


if __name__ == "__main__":
    main()
