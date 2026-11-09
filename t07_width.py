"""Test 7: how width changes what a network can fit.

A network with two hidden layers learns the two-spirals problem. Only the width
(neurons per hidden layer) changes, from 1 to 256. Each width is trained three
times with different random starts. The script prints training and validation
accuracy against the number of parameters and draws the decision boundary of a
narrow, a medium and a wide network.

Run:
    python t07_width.py            (about a minute)
    python t07_width.py --quick    (one seed, fewer widths)

Look for: very narrow networks cannot bend their boundary enough (underfitting);
accuracy climbs steeply as width grows, then levels off. Past that point extra
width buys almost nothing on this problem but costs parameters.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import count_params, make_spirals, plot_decision_boundary, results_dir, set_seed


def mlp(width):
    return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 2))


def fit(width, Xtr, ytr, seed, steps=1500):
    set_seed(seed)
    net = mlp(width)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(steps):  # full-batch training: the data set is small
        loss = F.cross_entropy(net(Xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def acc(net, X, y):
    with torch.no_grad():
        return (net(X).argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    widths = [1, 2, 4, 8, 16, 64, 256] if args.quick else [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256]
    seeds = [0] if args.quick else [0, 1, 2]

    Xtr, ytr = make_spirals(n=300, noise=0.2, seed=0)     # small training set
    Xva, yva = make_spirals(n=2000, noise=0.2, seed=1)    # fresh points from the same spirals
    Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))

    rows, nets = [], {}
    print(f"{'width':>5s} {'params':>8s} {'train acc':>10s} {'val acc':>8s} {'val range':>14s}")
    for w in widths:
        tr, va = [], []
        for s in seeds:
            net = fit(w, Xtr, ytr, seed=s)
            tr.append(acc(net, Xtr, ytr))
            va.append(acc(net, Xva, yva))
            if s == seeds[0]:
                nets[w] = net
        p = count_params(mlp(w))
        rows.append((w, p, np.mean(tr), np.mean(va), min(va), max(va)))
        print(f"{w:5d} {p:8,d} {np.mean(tr):10.1%} {np.mean(va):8.1%}   {min(va):5.1%} to {max(va):5.1%}")

    rows = np.array(rows)
    fig = plt.figure(figsize=(6.4, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.semilogx(rows[:, 1], rows[:, 2] * 100, "o-", ms=3.5, color="#3B6EA8", label="training accuracy")
    ax.semilogx(rows[:, 1], rows[:, 3] * 100, "o-", ms=3.5, color="#E07B39", label="validation accuracy")
    ax.fill_between(rows[:, 1], rows[:, 4] * 100, rows[:, 5] * 100, color="#E07B39", alpha=0.15)
    for w, p, _, va, _, _ in rows:
        if int(w) in (1, 2, 4, 8, 16, 64, 256):
            ax.annotate(f"w={int(w)}", (p, va * 100), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color="#555555")
    ax.set_xlabel("parameters (log scale)", fontsize=9)
    ax.set_ylabel("accuracy (%)", fontsize=9)
    ax.set_ylim(40, 103)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    show = [w for w in (2, 8, 128) if w in nets]
    for k, w in enumerate(show):
        a = fig.add_subplot(gs[1, k])
        net = nets[w]
        plot_decision_boundary(lambda g: net(torch.from_numpy(g)).argmax(1).numpy(),
                               Xtr.numpy(), ytr.numpy(), a,
                               title=f"width {w}: {count_params(net):,} params")
    fig.tight_layout()
    out = results_dir() / "t07_width.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
