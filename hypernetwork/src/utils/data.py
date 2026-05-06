"""
data.py
=======
Generates a family of regression tasks that the hypernetwork must learn to solve.

Task family: y = A * sin(ω * x + φ)
---------------------------------------
Each task is parameterized by:
  ω (frequency)  ∈ [freq_range[0], freq_range[1]]
  φ (phase)      ∈ [0, 2π]
  A (amplitude)  ∈ [amp_range[0], amp_range[1]]  (optional, default 1.0)

The task embedding fed to the hypernetwork is z = [sin(φ), cos(φ), ω, A]
(using sin/cos of phase to avoid the discontinuity at 2π→0).
"""

import torch
import math
from typing import Tuple


# ──────────────────────────────────────────────
# Single task
# ──────────────────────────────────────────────

def sample_task(
    freq_range: Tuple[float, float] = (0.5, 3.0),
    amp_range:  Tuple[float, float] = (0.5, 1.5),
) -> dict:
    """
    Sample a single sine task at random.

    Returns a dict with:
        z:    task embedding tensor of shape (4,)
        freq, phase, amp: the raw task parameters (for reference / plotting)
    """
    freq  = torch.empty(1).uniform_(*freq_range).item()
    phase = torch.empty(1).uniform_(0, 2 * math.pi).item()
    amp   = torch.empty(1).uniform_(*amp_range).item()

    z = torch.tensor([
        math.sin(phase),
        math.cos(phase),
        freq,
        amp,
    ])

    return dict(z=z, freq=freq, phase=phase, amp=amp)


def task_fn(x: torch.Tensor, freq: float, phase: float, amp: float) -> torch.Tensor:
    """Ground-truth output for a sine task."""
    return amp * torch.sin(freq * x + phase)


# ──────────────────────────────────────────────
# Batched task support for efficient training
# ──────────────────────────────────────────────

def sample_task_batch(
    batch_size: int,
    n_points:   int   = 50,
    x_range:    Tuple[float, float] = (-math.pi, math.pi),
    freq_range: Tuple[float, float] = (0.5, 3.0),
    amp_range:  Tuple[float, float] = (0.5, 1.5),
) -> dict:
    """
    Sample a batch of tasks and return (x, y, z) tensors ready for training.

    Returns:
        x:  (batch_size, n_points, 1) — input coordinates
        y:  (batch_size, n_points, 1) — ground-truth outputs
        z:  (batch_size, 4)           — task embeddings
        meta: list of per-task dicts with freq/phase/amp for plotting
    """
    xs, ys, zs, meta = [], [], [], []

    for _ in range(batch_size):
        task = sample_task(freq_range, amp_range)

        x = torch.linspace(*x_range, n_points).unsqueeze(-1)   # (n, 1)
        y = task_fn(x.squeeze(), task["freq"], task["phase"], task["amp"]).unsqueeze(-1)

        xs.append(x)
        ys.append(y)
        zs.append(task["z"])
        meta.append({k: task[k] for k in ("freq", "phase", "amp")})

    return dict(
        x    = torch.stack(xs),   # (B, n, 1)
        y    = torch.stack(ys),   # (B, n, 1)
        z    = torch.stack(zs),   # (B, 4)
        meta = meta,
    )
