"""Test 5: the learning rate.

Part A runs plain gradient descent on a bowl-shaped loss you can see completely,
L(w) = 0.5 * (1 * w1^2 + 10 * w2^2). Its curvature is 1 along w1 and 10 along w2.
Theory says gradient descent converges only if the learning rate is below
2 / (largest curvature) = 0.2. Four learning rates are tried: 0.02, 0.1, 0.19 and 0.21.

Part B trains a small network (2 -> 64 -> 64 -> 2, ReLU) on the spirals data with plain
mini-batch SGD at six learning rates from 0.001 to 10 and records the loss after each epoch.

Run:
    python t05_learning_rate.py

What to look for: too small a rate crawls, a good rate drops quickly, a rate near the
limit zig-zags or spikes, and above it training either stalls at chance or the loss
becomes NaN (not a number).
"""
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mlguide_data import make_spirals, results_dir, set_seed

COLS_A = ["#3B6EA8", "#4C9A5F", "#E07B39", "#C44E52"]           # too small, right, zig-zag, diverge
COLS_B = ["#7F7F7F", "#3B6EA8", "#4C9A5F", "#E07B39", "#8172B2", "#C44E52"]
CURV = np.array([1.0, 10.0])                     # curvature along w1 and w2


def bowl(w):
    return 0.5 * np.sum(CURV * w ** 2)


def gd_path(lr, w0=(-4.0, 1.5), steps=25):
    w = np.array(w0)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * CURV * w                     # gradient of the bowl is CURV * w
        path.append(w.copy())
    return np.array(path)


def train_spirals(lr, epochs=60, batch=32):
    set_seed(0)
    X, y = make_spirals(n=600, noise=0.2, seed=0)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            full = nn.functional.cross_entropy(model(X), y).item()
        losses.append(full)
        if not math.isfinite(full) or full > 1e3:  # blew up: stop early
            break
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item() if math.isfinite(losses[-1]) else float("nan")
    return losses, acc


def main():
    t0 = time.time()
    # ---- Part A ---------------------------------------------------------------------
    rates_a = [0.02, 0.1, 0.19, 0.21]
    print("Part A: gradient descent on L = 0.5*(w1^2 + 10*w2^2), 25 steps from (-4, 1.5)")
    print(f"{'learning rate':>13s} {'factor on w1':>13s} {'factor on w2':>13s} {'loss after 25':>14s}")
    paths = {}
    for lr in rates_a:
        p = gd_path(lr)
        paths[lr] = p
        print(f"{lr:13.2f} {1 - lr * CURV[0]:13.2f} {1 - lr * CURV[1]:13.2f} {bowl(p[-1]):14.4g}")

    # ---- Part B ---------------------------------------------------------------------
    rates_b = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0]
    print("\nPart B: 2-64-64-2 ReLU network on spirals, plain SGD, batch 32, 60 epochs")
    print(f"{'learning rate':>13s} {'loss after 1 epoch':>19s} {'final loss':>11s} {'accuracy':>9s}")
    curves = {}
    for lr in rates_b:
        losses, acc = train_spirals(lr)
        curves[lr] = losses
        final = losses[-1]
        status = f"{final:11.4f}" if math.isfinite(final) and final < 1e3 else f"{'diverged':>11s}"
        accs = f"{acc:9.3f}" if math.isfinite(acc) else f"{'-':>9s}"
        print(f"{lr:13.3f} {losses[0]:19.4f} {status} {accs}")

    # ---- Figure ------------------------------------------------------------------------
    fig = plt.figure(figsize=(9.6, 6.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    g1, g2 = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-3, 3, 200))
    zz = 0.5 * (CURV[0] * g1 ** 2 + CURV[1] * g2 ** 2)
    names = {0.02: "0.02: too small", 0.1: "0.1: about right", 0.19: "0.19: zig-zags", 0.21: "0.21: diverges"}
    for k, lr in enumerate(rates_a):
        ax = fig.add_subplot(gs[0, k])
        ax.contour(g1, g2, zz, levels=np.geomspace(0.05, 60, 12), colors="#BBBBBB", linewidths=0.7)
        p = paths[lr]
        ax.plot(p[:, 0], p[:, 1], "-o", color=COLS_A[k], ms=2.5, lw=1.2)
        ax.plot(0, 0, "k*", ms=8)
        ax.set_xlim(-5, 5); ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step size {names[lr]}", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    for k, lr in enumerate(rates_b):
        ls = np.array(curves[lr])
        if not (math.isfinite(ls[-1]) and ls[-1] < 1e3):    # blew up: mark it at the top edge
            ax.plot([1], [8.0], marker="X", ms=10, color=COLS_B[k], ls="none",
                    label=f"lr = {lr:g}: loss became NaN in epoch {len(ls)}")
            continue
        ax.plot(np.arange(1, len(ls) + 1), ls, color=COLS_B[k], lw=2, label=f"lr = {lr:g}")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 12)
    ax.set_xlabel("epoch", fontsize=9); ax.set_ylabel("training loss (log scale)", fontsize=9)
    ax.set_title("A small network on spirals, plain SGD at six learning rates", fontsize=9.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t05_learning_rate.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
