"""
training.py
===========
End-to-end training loop for the hypernetwork + target network system.

The key insight is that **both forward passes are differentiable**:
  loss = MSE( target_network(x, hyper_network(z)), y )

Gradients flow back through the target network's computation, then through
the generated weights, and finally into the hypernetwork's parameters.
No separate optimisation is needed for the target network.
"""

import torch
import torch.nn as nn
from typing import Callable

from src.hypernetwork.model  import HyperNetwork
from src.target_network.model import TargetNetwork
from src.utils.data           import sample_task_batch


def train(
    hyper_net:   HyperNetwork,
    target_net:  TargetNetwork,
    n_steps:     int   = 2000,
    batch_size:  int   = 16,
    n_points:    int   = 50,
    lr:          float = 1e-3,
    log_every:   int   = 200,
    on_log:      Callable | None = None,
) -> list:
    """
    Train the hypernetwork to generate weights for the target network.

    Args:
        hyper_net:   The HyperNetwork to train.
        target_net:  A stateless TargetNetwork (no parameters of its own).
        n_steps:     Total number of gradient steps.
        batch_size:  Number of tasks sampled per step.
        n_points:    Number of (x, y) pairs per task.
        lr:          Learning rate for AdamW.
        log_every:   Print a status line every this many steps.
        on_log:      Optional callback(step, loss) for custom logging.

    Returns:
        history: list of (step, loss) tuples recorded every `log_every` steps.
    """
    optimizer = torch.optim.AdamW(hyper_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history   = []

    hyper_net.train()

    for step in range(1, n_steps + 1):
        # ── 1. Sample a batch of tasks ──────────────────────────────────
        batch = sample_task_batch(batch_size, n_points)
        x = batch["x"]   # (B, n, 1)
        y = batch["y"]   # (B, n, 1)
        z = batch["z"]   # (B, 4)

        # ── 2. HyperNetwork forward: z → θ ──────────────────────────────
        params = hyper_net(z)  # list of (W, b) pairs, each with batch dim

        # ── 3. Target network forward for every task in the batch ────────
        # params[l] has shape (B, out, in) for W and (B, out) for b.
        # We evaluate each task's target network separately and stack.
        preds = []
        for b_idx in range(batch_size):
            task_params = [(W[b_idx], b_vec[b_idx]) for W, b_vec in params]
            pred = target_net(x[b_idx], task_params)   # (n, 1)
            preds.append(pred)

        preds = torch.stack(preds)   # (B, n, 1)

        # ── 4. Loss + backprop ───────────────────────────────────────────
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(hyper_net.parameters(), max_norm=1.0)

        optimizer.step()

        # ── 5. Logging ───────────────────────────────────────────────────
        if step % log_every == 0 or step == 1:
            loss_val = loss.item()
            history.append((step, loss_val))
            print(f"Step {step:>5d} / {n_steps}   loss = {loss_val:.6f}")
            if on_log:
                on_log(step, loss_val)

    return history


@torch.no_grad()
def evaluate_task(
    hyper_net:  HyperNetwork,
    target_net: TargetNetwork,
    z:          torch.Tensor,
    x:          torch.Tensor,
) -> torch.Tensor:
    """
    Run a single task through the hypernetwork + target network.

    Args:
        z: Task embedding of shape (4,) or (1, 4).
        x: Input points of shape (n, 1) or (1, n, 1).

    Returns:
        Predicted output of shape (n, 1).
    """
    hyper_net.eval()

    if z.dim() == 1:
        z = z.unsqueeze(0)   # (1, 4)

    params = hyper_net(z)                             # list of (1, out, in)
    task_params = [(W[0], b[0]) for W, b in params]   # strip batch dim

    if x.dim() == 1:
        x = x.unsqueeze(-1)

    return target_net(x, task_params)
