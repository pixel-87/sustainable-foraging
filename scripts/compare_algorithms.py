#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_run(run_dir: Path):
    csv_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "config.json"
    if not csv_path.exists(): return None

    config = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)

    ts, rew, food, leng, rem = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts.append(int(float(row.get("timestep", 0) or 0)))
            rew.append(float(row.get("reward_total", 0) or 0))
            food.append(float(row.get("foods_collected", 0) or 0))
            leng.append(float(row.get("length", 0) or 0))
            rem.append(float(row.get("food_remaining_end", 0) or 0))

    if not ts: return None

    ts = np.array(ts)
    rew = np.array(rew)
    food = np.array(food)
    leng = np.array(leng, dtype=float)
    rem = np.array(rem)

    # Label logic
    lib = config.get("library", "")
    algo = config.get("algorithm", "")
    if lib and algo:
        label = f"{lib.upper()}/{algo.upper()}"
    elif algo:
        label = algo.upper()
    else:
        label = run_dir.name.split("_")[0].upper()

    eff = leng / np.maximum(food, 1.0)

    return {
        "run_dir": run_dir,
        "label": label,
        "ts": ts,
        "reward": rew,
        "length": leng,
        "sustainability": rem,
        "restraint": eff
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", default=[])
    args = ap.parse_args()

    dirs = [Path(d) for d in args.run_dirs]
    if not dirs:
        dirs = sorted(d for d in Path("logs").glob("*_pomdp_seed*") if (d / "metrics.csv").exists())

    runs = [r for d in dirs if (r := load_run(d)) is not None]
    if not runs:
        print("No valid run data found.")
        sys.exit(1)

    # Group by label
    groups = defaultdict(list)
    for r in runs:
        groups[r["label"]].append(r)

    max_global_ts = max(max(r["ts"]) for r in runs)

    # Prepare plots
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.get_cmap("tab10").colors

    fig_all, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig_len, ax_len = plt.subplots(figsize=(10, 6))
    fig_sus, ax_sus = plt.subplots(figsize=(10, 6))
    fig_rew, ax_rew = plt.subplots(figsize=(10, 6))

    metric_specs = [
        ("length", "Episode Length", "Steps", axes[0], ax_len),
        ("sustainability", "Sustainability", "Food Remaining", axes[1], ax_sus),
        ("reward", "Episodic Reward", "Reward Total", axes[2], ax_rew)
    ]

    def format_axes(ax, title, ylabel):
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1e6:g}M"))
        ax.grid(True, alpha=0.3)

    for m_key, m_title, m_ylabel, ax_sub, ax_ind in metric_specs:
        format_axes(ax_sub, m_title, m_ylabel)
        format_axes(ax_ind, m_title, m_ylabel)

    # To store final stats for leaderboard
    leaderboard_data = []

    # Sort groups alphabetically for consistent colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'X', '*', '+']
    for idx, (label, group_runs) in enumerate(sorted(groups.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # Find max ts for this specific algorithm
        algo_max_ts = max(max(r["ts"]) for r in group_runs)

        # Interpolate onto shared evenly spaced x-axis for this algorithm's duration
        algo_x = np.linspace(0, algo_max_ts, 1000)

        final_stats = {"label": label}

        for m_key, m_title, m_ylabel, ax_sub, ax_ind in metric_specs:
            interp_lines = []
            for r in group_runs:
                interp_y = np.interp(algo_x, r["ts"], r[m_key])
                interp_lines.append(interp_y)

            mean_y = np.mean(interp_lines, axis=0)
            std_y = np.std(interp_lines, axis=0)

            # Use Standard Error of the Mean (SEM) to make bands tighter
            n_runs = len(group_runs)
            err_y = std_y / np.sqrt(n_runs) if n_runs > 0 else np.zeros_like(std_y)

            # Smooth the mean and std slightly for visual clarity (window=25 out of 1000)
            def smooth(v, w=25):
                pad = w//2
                pv = np.pad(v, (pad, pad), mode='edge')
                # np.convolve extends length by w-1, so we slice to match original length
                res = np.convolve(pv, np.ones(w)/w, mode='valid')
                # handle off-by-one caused by even/odd padding lengths
                if len(res) > len(v): res = res[:len(v)]
                elif len(res) < len(v): res = np.pad(res, (0, len(v)-len(res)), mode='edge')
                return res

            mean_smooth = smooth(mean_y)
            err_smooth = smooth(err_y)

            # Bin the data into 50 discrete points to average out spikes
            n_bins = 50
            algo_x_binned = algo_x.reshape(n_bins, -1).mean(axis=1)
            mean_binned = mean_smooth.reshape(n_bins, -1).mean(axis=1)
            err_binned = err_smooth.reshape(n_bins, -1).mean(axis=1)

            # Plot on both sub and ind
            is_baseline = "BASELINE" in label.upper()
            line_style = "--" if is_baseline else "-"
            line_marker = None if is_baseline else marker
            line_alpha = 0.5 if is_baseline else 1.0

            for ax in [ax_sub, ax_ind]:
                ax.plot(algo_x_binned, mean_binned, color=color, label=label, lw=2 if is_baseline else 1.5, ls=line_style, marker=line_marker, markersize=4, alpha=line_alpha)

                # Only shade the standard error if it's not a baseline (baselines are flat averages here anyway)
                if not is_baseline:
                    fill_alpha = 0.15 if ax == ax_sub else 0.25
                    ax.fill_between(algo_x_binned, mean_binned - err_binned, mean_binned + err_binned, color=color, alpha=fill_alpha)

                # Extension dashed line
                if algo_max_ts < max_global_ts:
                    ax.plot([algo_max_ts, max_global_ts], [mean_binned[-1], mean_binned[-1]], color=color, ls="--", lw=1.5, alpha=0.4)

            final_stats[m_key] = (mean_binned[-1], err_binned[-1])

        leaderboard_data.append(final_stats)

    # Add legends and layout
    handles, labels = axes[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc='lower center', ncol=len(groups), bbox_to_anchor=(0.5, 0.02))

    for m_key, m_title, m_ylabel, ax_sub, ax_ind in metric_specs:
        ax_ind.legend(loc='best')

    fig_all.tight_layout(rect=[0, 0.08, 1, 1])
    fig_len.tight_layout()
    fig_sus.tight_layout()
    fig_rew.tight_layout()

    out = Path("logs")

    # Save the uncropped versions for the appendix
    fig_all.savefig(out / "comparison_all_appendix.png", dpi=200)
    fig_len.savefig(out / "comparison_length_appendix.png", dpi=200)
    fig_sus.savefig(out / "comparison_sustainability_appendix.png", dpi=200)
    fig_rew.savefig(out / "comparison_reward_appendix.png", dpi=200)

    # Crop the x-axis to 2M for the main figures
    for m_key, m_title, m_ylabel, ax_sub, ax_ind in metric_specs:
        ax_sub.set_xlim(0, 2_000_000)
        ax_ind.set_xlim(0, 2_000_000)

    fig_all.savefig(out / "comparison_all.png", dpi=200)
    fig_len.savefig(out / "comparison_length.png", dpi=200)
    fig_sus.savefig(out / "comparison_sustainability.png", dpi=200)
    fig_rew.savefig(out / "comparison_reward.png", dpi=200)

    # Leaderboard
    leaderboard_data.sort(key=lambda x: x["length"][0], reverse=True)

    print("\n### Final Algorithm Benchmark Leaderboard\n")
    print("| Algorithm | Episode Length (Mean ± SEM) | Sustainability (Mean ± SEM) | Episodic Reward (Mean ± SEM) |")
    print("| :--- | :--- | :--- | :--- |")
    for row in leaderboard_data:
        len_m, len_s = row["length"]
        sus_m, sus_s = row["sustainability"]
        rew_m, rew_s = row["reward"]
        print(f"| **{row['label']}** | {len_m:.2f} ± {len_s:.2f} | {sus_m:.2f} ± {sus_s:.2f} | {rew_m:.2f} ± {rew_s:.2f} |")
    print("\nSaved graphs to logs/comparison_all.png, logs/comparison_length.png, logs/comparison_sustainability.png, logs/comparison_reward.png\n")

    # Export to CSV and LaTeX for dissertation
    with open(out / "dissertation_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Episode Length (Mean)", "Episode Length (SEM)", "Sustainability (Mean)", "Sustainability (SEM)", "Episodic Reward (Mean)", "Episodic Reward (SEM)"])
        for row in leaderboard_data:
            writer.writerow([row['label'], f"{row['length'][0]:.2f}", f"{row['length'][1]:.2f}", f"{row['sustainability'][0]:.2f}", f"{row['sustainability'][1]:.2f}", f"{row['reward'][0]:.2f}", f"{row['reward'][1]:.2f}"])

    with open(out / "dissertation_table.typ", "w") as f:
        f.write("#figure(\n")
        f.write("  table(\n")
        f.write("    columns: 4,\n")
        f.write("    align: (left, center, center, center),\n")
        f.write("    [*Algorithm*], [*Episode Length*], [*Sustainability*], [*Episodic Reward*],\n")
        for row in leaderboard_data:
            label = row['label'].replace('_', '\\_')
            len_str = f"{row['length'][0]:.2f} \\pm {row['length'][1]:.2f}"
            sus_str = f"{row['sustainability'][0]:.2f} \\pm {row['sustainability'][1]:.2f}"
            rew_str = f"{row['reward'][0]:.2f} \\pm {row['reward'][1]:.2f}"
            f.write(f"    [{label}], [${len_str}$], [${sus_str}$], [${rew_str}$],\n")
        f.write("  ),\n")
        f.write("  caption: [Final benchmark results showing Mean $\\pm$ SEM across all seeds.],\n")
        f.write(") <tab:benchmark_results>\n")

    print("Saved dissertation tables to logs/dissertation_table.csv and logs/dissertation_table.typ\n")

if __name__ == "__main__":
    main()
