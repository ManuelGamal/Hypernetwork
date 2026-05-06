"""
train.py
========
Entry point.  Trains the hypernetwork and saves plots.

Usage
-----
    python train.py [--steps 2000] [--batch-size 16] [--lr 1e-3] [--out ./outputs]
"""

import argparse
import math
import os
import torch

from src.hypernetwork.model   import HyperNetwork
from src.target_network.model import TargetNetwork
from src.utils.training       import train, evaluate_task
from src.utils.data           import sample_task, task_fn
from src.visualization.plot   import (
    plot_loss_curve,
    plot_task_fit,
    plot_weight_grid,
    plot_generalization,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a hypernetwork on the sine task family.")
    p.add_argument("--steps",       type=int,   default=2000,  help="Training steps")
    p.add_argument("--batch-size",  type=int,   default=16,    help="Tasks per step")
    p.add_argument("--lr",          type=float, default=1e-3,  help="Learning rate")
    p.add_argument("--hidden-dim",  type=int,   default=128,   help="HyperNet hidden size")
    p.add_argument("--out",         type=str,   default="outputs", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── Architecture ──────────────────────────────────────────────────────
    # Target network: 1 → 32 → 32 → 1
    target_net = TargetNetwork(layer_sizes=[1, 32, 32, 1], activation="tanh")
    print(f"Target network: {target_net.layer_sizes}")
    print(f"  Total params to generate: {target_net.n_params}")

    # HyperNetwork: z (dim=4) → hidden → generated weights
    hyper_net = HyperNetwork(
        z_dim         = 4,
        hidden_dim    = args.hidden_dim,
        target_shapes = target_net.weight_shapes,
    )
    n_hyper = sum(p.numel() for p in hyper_net.parameters())
    print(f"HyperNetwork hidden_dim={args.hidden_dim}")
    print(f"  HyperNet params: {n_hyper:,}")
    print(f"  Parameter ratio: {n_hyper / target_net.n_params:.1f}x\n")

    # ── Training ──────────────────────────────────────────────────────────
    print("Training …")
    history = train(
        hyper_net   = hyper_net,
        target_net  = target_net,
        n_steps     = args.steps,
        batch_size  = args.batch_size,
        lr          = args.lr,
    )

    # ── Save model ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.out, "hyper_net.pt")
    torch.save(hyper_net.state_dict(), ckpt_path)
    print(f"\nModel saved → {ckpt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("Generating plots …")

    plot_loss_curve(history).savefig(
        os.path.join(args.out, "loss_curve.png"), dpi=150, bbox_inches="tight"
    )

    # A few task fits (random tasks not seen during training)
    for i in range(3):
        task = sample_task()
        plot_task_fit(hyper_net, target_net, task).savefig(
            os.path.join(args.out, f"task_fit_{i}.png"), dpi=150, bbox_inches="tight"
        )

    plot_weight_grid(hyper_net).savefig(
        os.path.join(args.out, "weight_distributions.png"), dpi=150, bbox_inches="tight"
    )

    plot_generalization(hyper_net, target_net).savefig(
        os.path.join(args.out, "generalization_grid.png"), dpi=150, bbox_inches="tight"
    )

    print(f"\nAll plots saved to ./{args.out}/")
    print("Done.")


if __name__ == "__main__":
    main()
