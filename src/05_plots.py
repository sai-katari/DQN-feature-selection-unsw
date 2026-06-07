# 05_plots.py - generate training curves and results comparison plots

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def moving_average(x, w=30):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load():
    with open(C.METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)
    with open(os.path.join(C.RESULTS_DIR, "final_results.pkl"), "rb") as f:
        final = pickle.load(f)
    return metrics, final


def plot_learning_curves(metrics, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("DQN Feature Selection - Training Curves (UNSW-NB15)", fontsize=13, fontweight="bold")

    n_ep = len(metrics["episode_rewards"])
    eps  = np.arange(1, n_ep + 1)
    w    = 30

    def plot_with_smooth(ax, data, color, ylabel, title):
        ax.plot(eps, data, color=color, alpha=0.25, linewidth=0.7)
        if len(data) >= w:
            smooth = moving_average(data, w)
            ax.plot(eps[w-1:], smooth, color=color, linewidth=1.8, label=f"MA-{w}")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_with_smooth(axes[0], metrics["episode_rewards"], "#2563EB", "Cumulative Reward", "Episode Reward")
    plot_with_smooth(axes[1], metrics["episode_f1"],      "#16A34A", "Macro F1",           "Validation Macro F1")
    plot_with_smooth(axes[2], metrics["episode_nfeat"],   "#DC2626", "# Features",         "Features Selected")

    axes[3].plot(eps, metrics["episode_epsilon"], color="#7C3AED", linewidth=1.5)
    axes[3].set_ylabel("Epsilon", fontsize=9)
    axes[3].set_title("Exploration Rate", fontsize=9, pad=3)
    axes[3].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Episode", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_results_bar(final_results, out_path):
    methods = list(final_results.keys())
    f1s     = [final_results[m]["macro_f1"]  for m in methods]
    nfeats  = [final_results[m]["n_features"] for m in methods]
    colors  = ["#94A3B8"] * (len(methods) - 1) + ["#2563EB"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, f1s, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001, f"{f1:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for bar, nf in zip(bars, nfeats):
        ax.text(bar.get_x() + bar.get_width() / 2,
                0.841, f"k={nf}",
                ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.84, max(f1s) + 0.015)
    ax.set_ylabel("Macro F1 (Test Set)", fontsize=11)
    ax.set_title("Feature Selection Methods - UNSW-NB15", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def plot_feature_histogram(metrics, out_path):
    nfeats = metrics["episode_nfeat"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nfeats, bins=30, color="#2563EB", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(np.median(nfeats), color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"median={np.median(nfeats):.0f}")
    ax.set_xlabel("features selected per episode", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.set_title("Feature Count Distribution Across Training Episodes", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def main():
    metrics, final_results = load()
    plot_learning_curves(metrics,   os.path.join(C.RESULTS_DIR, "learning_curves.png"))
    plot_results_bar(final_results, os.path.join(C.RESULTS_DIR, "results_comparison.png"))
    plot_feature_histogram(metrics, os.path.join(C.RESULTS_DIR, "feature_count_dist.png"))
    print("all plots saved to results/")


if __name__ == "__main__":
    main()