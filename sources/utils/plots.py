import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#b5b3ac"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def number(value, sign=False):
    return f"{value:{'+' if sign else ''}.1f}"


def style(ax, labelsize=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=labelsize)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def axes(height):
    fig, ax = plt.subplots(figsize=(9, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    return fig, ax


def plot_results(results, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(results["CLIP"])
    clip = np.array([results["CLIP"][m] for m in metrics])
    runs = np.array([[results[s][m] for m in metrics] for s in seeds])
    mean, std = runs.mean(0), runs.std(0, ddof=1)
    y = np.arange(len(metrics))[::-1]

    fig, ax = axes(0.45 * len(metrics) + 1.2)
    ax.barh(y + 0.19, clip, height=0.36, color=GRAY, label="CLIP (frozen)")
    ax.barh(y - 0.19, mean, height=0.36, color=BLUE, xerr=std, ecolor=INK, error_kw={"linewidth": 1, "capsize": 2},
            label="NegSplit (mean ± std, 4 seeds)")
    for yi, c, m, s in zip(y, clip, mean, std):
        ax.text(c + 0.8, yi + 0.19, number(c), va="center", fontsize=7, color=MUTED)
        ax.text(m + s + 0.8, yi - 0.19, number(m), va="center", fontsize=7, color=INK)
    ax.set_yticks(y, metrics, fontsize=8, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("accuracy / R@k (%)", color=MUTED, fontsize=8)
    ax.set_title("CLIP and NegSplit on the test sets", loc="left", color=INK, fontsize=11, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", facecolor=SURFACE)
    plt.close(fig)

    delta = mean - clip
    fig, ax = axes(0.38 * len(metrics) + 1.2)
    ax.barh(y, delta, height=0.6, color=[BLUE if d >= 0 else ORANGE for d in delta])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    for yi, d in zip(y, delta):
        ax.text(d + (0.8 if d >= 0 else -0.8), yi, number(d, sign=True), va="center",
                ha="left" if d >= 0 else "right", fontsize=7, color=INK)
    ax.set_yticks(y, metrics, fontsize=8, color=INK)
    ax.set_xlim(min(delta.min(), 0) - 6, delta.max() + 8)
    ax.set_xlabel("change vs. frozen CLIP (pp)", color=MUTED, fontsize=8)
    ax.set_title("What NegSplit changes vs. CLIP (mean of 4 seeds)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "change_vs_clip.png", facecolor=SURFACE)
    plt.close(fig)
