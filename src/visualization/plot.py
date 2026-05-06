"""
plot.py
=======
Visualisation helpers for the hypernetwork demo.

Functions
---------
plot_loss_curve      — training loss over steps
plot_task_fit        — predicted vs ground-truth curve for one task
plot_weight_grid     — histogram of generated weights for several tasks
plot_generalization  — grid of predicted curves across (freq, phase) space
"""

import math
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Tuple

from src.hypernetwork.model   import HyperNetwork
from src.target_network.model import TargetNetwork
from src.utils.data           import task_fn, sample_task
from src.utils.training       import evaluate_task


# ── Shared style ─────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor":  "#0e1018",
    "axes.facecolor":    "#0e1018",
    "axes.edgecolor":    "#2a2d3e",
    "text.color":        "#d4d8e8",
    "axes.labelcolor":   "#d4d8e8",
    "xtick.color":       "#5a6080",
    "ytick.color":       "#5a6080",
    "grid.color":        "#1e2130",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}

ACCENT   = "#00ffa3"
ACCENT2  = "#ff6b35"
ACCENT3  = "#7c6aff"
MUTED    = "#5a6080"


def _apply_style():
    plt.rcParams.update(STYLE)


# ── 1. Loss curve ─────────────────────────────────────────────────────────────

def plot_loss_curve(history: List[Tuple[int, float]], save_path: str = None):
    """Plot MSE loss over training steps."""
    _apply_style()
    steps, losses = zip(*history)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, losses, color=ACCENT, linewidth=2, label="MSE loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("HyperNetwork Training Loss", color="#d4d8e8")
    ax.legend(framealpha=0.2)
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 2. Single task fit ────────────────────────────────────────────────────────

def plot_task_fit(
    hyper_net:  HyperNetwork,
    target_net: TargetNetwork,
    task:       dict = None,
    save_path:  str  = None,
):
    """
    Show predicted vs ground-truth curve for one task.

    Args:
        task: dict with keys z, freq, phase, amp.  Sampled at random if None.
    """
    _apply_style()
    if task is None:
        task = sample_task()

    x   = torch.linspace(-math.pi, math.pi, 200).unsqueeze(-1)
    y   = task_fn(x.squeeze(), task["freq"], task["phase"], task["amp"])
    y_hat = evaluate_task(hyper_net, target_net, task["z"], x).squeeze().detach()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x.squeeze(), y,     color=MUTED,   linewidth=2, label="Ground truth", linestyle="--")
    ax.plot(x.squeeze(), y_hat, color=ACCENT,  linewidth=2, label="HyperNet prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Task fit   ω={task['freq']:.2f}  φ={task['phase']:.2f}  A={task['amp']:.2f}",
        color="#d4d8e8",
    )
    ax.legend(framealpha=0.2)
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 3. Weight distribution grid ──────────────────────────────────────────────

def plot_weight_grid(
    hyper_net: HyperNetwork,
    n_tasks:   int = 6,
    save_path: str = None,
):
    """
    For each of n_tasks randomly sampled tasks, plot the histogram of
    generated weights.  Illustrates that the hypernetwork produces
    *different* weight distributions for different tasks.
    """
    _apply_style()
    tasks  = [sample_task() for _ in range(n_tasks)]
    z_batch = torch.stack([t["z"] for t in tasks])

    with torch.no_grad():
        params = hyper_net(z_batch)           # list of (n_tasks, out, in)

    all_weights = []
    for W, b in params:
        all_weights.append(W.reshape(n_tasks, -1))
        all_weights.append(b.reshape(n_tasks, -1))
    all_weights = torch.cat(all_weights, dim=1).numpy()  # (n_tasks, total_params)

    cols = 3
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n_tasks]):
        ax.hist(all_weights[i], bins=30, color=ACCENT3, alpha=0.8, edgecolor="none")
        t = tasks[i]
        ax.set_title(
            f"ω={t['freq']:.1f}  φ={t['phase']:.1f}  A={t['amp']:.1f}",
            fontsize=9, color="#d4d8e8",
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y")

    for ax in axes[n_tasks:]:
        ax.set_visible(False)

    fig.suptitle("Generated Weight Distributions per Task", color="#d4d8e8", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 4. Generalization grid ────────────────────────────────────────────────────

def plot_generalization(
    hyper_net:  HyperNetwork,
    target_net: TargetNetwork,
    freq_vals:  List[float] = None,
    phase_vals: List[float] = None,
    save_path:  str = None,
):
    """
    Grid of (freq × phase) showing how well the hypernetwork generalizes
    across the task space.  Each cell shows predicted (green) vs true (dashed).
    """
    _apply_style()
    if freq_vals  is None: freq_vals  = [0.5, 1.0, 2.0, 3.0]
    if phase_vals is None: phase_vals = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    x = torch.linspace(-math.pi, math.pi, 200).unsqueeze(-1)

    rows, cols = len(phase_vals), len(freq_vals)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))

    for r, phase in enumerate(phase_vals):
        for c, freq in enumerate(phase_vals[:len(freq_vals)]):
            freq_val = freq_vals[c]
            task = dict(
                freq=freq_val, phase=phase, amp=1.0,
                z=torch.tensor([math.sin(phase), math.cos(phase), freq_val, 1.0]),
            )
            y     = task_fn(x.squeeze(), freq_val, phase, 1.0)
            y_hat = evaluate_task(hyper_net, target_net, task["z"], x).squeeze().detach()

            ax = axes[r][c]
            ax.plot(x.squeeze(), y,     color=MUTED,  linewidth=1.5, linestyle="--")
            ax.plot(x.squeeze(), y_hat, color=ACCENT, linewidth=1.5)
            ax.set_ylim(-2, 2)
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"ω={freq_val}", fontsize=9, color="#d4d8e8")
            if c == 0:
                ax.set_ylabel(f"φ={phase:.1f}", fontsize=9, color="#d4d8e8")

    fig.suptitle(
        "Generalization across task space\n(green = predicted, dashed = true)",
        color="#d4d8e8", y=1.02,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
