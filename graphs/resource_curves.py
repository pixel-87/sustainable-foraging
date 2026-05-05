import matplotlib.pyplot as plt
import numpy as np

K = 2
alpha = 1.6  # critical alpha from sustainable_benchmark.py Fair preset
F_min = 0.3  # from sustainable_benchmark.py Fair preset


def regrowth(r_t, foraged=0):
    """Logistic regrowth equation: r_{t+1} = alpha * r_t - (alpha - 1) / K * r_t^2 - foraged"""
    r_t = np.asarray(r_t)
    result = np.where(
        (K > 0) & (r_t > 0), alpha * r_t - (alpha - 1.0) / K * (r_t**2) - foraged, -foraged
    )
    return float(result) if result.ndim == 0 else result


def delta_r(r_t, foraged=0):
    """Net change in resource level (regrowth minus foraging)"""
    r_t = np.asarray(r_t)
    result = np.where(
        (K > 0) & (r_t > 0),
        alpha * r_t - (alpha - 1.0) / K * (r_t**2) - foraged - r_t,
        -foraged - r_t,
    )
    return float(result) if result.ndim == 0 else result


r = np.linspace(0, K, 100)

fig1, ax1 = plt.subplots(figsize=(10, 6))

r_t_values = [0.1]
for t in range(1, 51):
    r_next = regrowth(r_t_values[-1], foraged=0)
    r_next = max(0, min(r_next, K))
    r_t_values.append(r_next)
    if r_next >= K * 0.99:
        r_t_values.extend([K] * (50 - t))
        break

time_steps = np.arange(len(r_t_values))
ax1.plot(time_steps, r_t_values, "b-", linewidth=2.5, label="Natural Regrowth Curve")
ax1.axhline(y=K, color="gray", linestyle="--", alpha=0.5, label=f"Carrying Capacity K={K}")
ax1.axvline(x=len(r_t_values) - 1, color="gray", linestyle=":", alpha=0.5)

inflection_idx = np.argmin(np.abs(np.array(r_t_values) - 1.0))
ax1.scatter(
    [inflection_idx], [r_t_values[inflection_idx]], color="red", s=100, zorder=5, marker="o"
)
ax1.annotate(
    "Maximum Growth\nInflection Point\n($r_t = 1.0 = K/2$)",
    xy=(inflection_idx, r_t_values[inflection_idx]),
    xytext=(inflection_idx + 5, r_t_values[inflection_idx] + 0.3),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red",
)

ax1.set_xlabel("Time (Steps)", fontsize=12)
ax1.set_ylabel("Resource Level ($r_t$)", fontsize=12)
ax1.set_title(
    "Graph 1: The Natural Regrowth Curve\n(How the environment recovers when agents do nothing)",
    fontsize=14,
)
ax1.set_xlim(0, 15)
ax1.set_ylim(0, K + 0.1)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right")
plt.tight_layout()
plt.savefig(
    "/home/pixel/code/dissertation/sustainable-foraging/graphs/graph1_natural_regrowth.png", dpi=150
)
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 6))

deltas = delta_r(r, foraged=0)
ax2.plot(r, deltas, "b-", linewidth=2.5, label="Regrowth Parabola\n$\\Delta r = r_{t+1} - r_t$")
ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
ax2.axvline(x=0, color="gray", linestyle="-", alpha=0.5)
ax2.axvline(x=K, color="gray", linestyle=":", alpha=0.5, label=f"Carrying Capacity K={K}")

ax2.axhline(y=F_min, color="red", linestyle="--", linewidth=2, label=f"$F_{{min}} = {F_min}$")

peak_idx = np.argmax(deltas)
ax2.scatter([r[peak_idx]], [deltas[peak_idx]], color="red", s=100, zorder=5, marker="*")
ax2.annotate(
    f"Peak at $(r_t={r[peak_idx]:.2f}, \\Delta r={deltas[peak_idx]:.2f})$",
    xy=(r[peak_idx], deltas[peak_idx]),
    xytext=(r[peak_idx] - 0.5, deltas[peak_idx] + 0.08),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red",
)

r_critical = 1.0
delta_at_critical = delta_r(r_critical, foraged=0)
ax2.scatter([r_critical], [delta_at_critical], color="green", s=100, zorder=5, marker="s")
ax2.annotate(
    f"Critical Point $(r_t={r_critical}, \\Delta r={delta_at_critical:.2f})$",
    xy=(r_critical, delta_at_critical),
    xytext=(r_critical + 0.3, delta_at_critical - 0.1),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="green"),
    color="green",
)

ax2.set_xlabel("Current Resource Level ($r_t$)", fontsize=12)
ax2.set_ylabel(r"Expected Regrowth ($\Delta r = r_{t+1} - r_t$)", fontsize=12)
ax2.set_title(
    "Graph 2: The Regrowth Parabola\n(Why agents must keep food at exactly 1.0 to survive)",
    fontsize=14,
)
ax2.set_xlim(0, K)
ax2.set_ylim(-0.1, max(deltas) + 0.15)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

annotation_text = (
    "The peak of the parabola at $r_t = 1.0$ (K/2) yields maximum\n"
    f"regrowth of {delta_at_critical:.2f}. This exactly equals $F_min = {F_min}$,\n"
    "proving the critical threshold formula."
)
ax2.text(
    0.05,
    0.50,
    annotation_text,
    transform=ax2.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(
    "/home/pixel/code/dissertation/sustainable-foraging/graphs/graph2_regrowth_parabola.png",
    dpi=150,
)
plt.close()

print("Graphs saved:")
print("  - graph1_natural_regrowth.png")
print("  - graph2_regrowth_parabola.png")
print(f"\nParameters: K={K}, alpha={alpha}")
print(f"Peak regrowth at r_t=K/2: {delta_at_critical:.3f}")
print(f"F_min (Fair preset): {F_min}")
