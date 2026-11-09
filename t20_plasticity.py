"""Test 20: loss of plasticity, a network that slowly loses the ability to learn.

Forgetting is one half of the continual-learning problem. The other half is that a network
trained for a long time on one task after another can lose its ability to learn new ones
(Lyle et al. 2023; Dohare et al. 2024).

Here every task uses the same 1,000 digit images but gives them fresh random labels, so each
task is pure learning: nothing carries over from the last one. Every task gets the same
budget (250 Adam steps, with a fresh optimiser). We measure the training accuracy reached
by the end of each task, which is a direct measure of how well the network can still learn.

Three versions:
  standard           keep training the same network from task to task
  reset              start every task from a freshly initialised network (the reference)
  shrink & perturb   at each task boundary, halve the weights and add half of a fresh
                     random initialisation (after Ash and Adams 2020)

It also tracks three symptoms that Dohare et al. (2024) found alongside plasticity loss:
the share of dead units in the first hidden layer (units that output zero for every image),
the effective rank of the last hidden layer's features (roughly, how many independent
directions the network's features still use) and the overall size of the weights.

Run:   python t20_plasticity.py            (about 2 minutes)
       python t20_plasticity.py --quick    (12 tasks instead of 25)
Saves: results/t20_plasticity.png
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

from mlguide_data import synthetic_digits, set_seed, results_dir

N_IMAGES = 1000
STEPS_PER_TASK = 250
BATCH = 100


def make_net():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def effective_rank(H):
    """exp(entropy of the normalised singular values): about how many directions H uses."""
    s = torch.linalg.svdvals(H - H.mean(0))
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


@torch.no_grad()
def symptoms(net, X):
    """Dead units in hidden layer 1, effective rank of hidden layer 2, size of the weights."""
    h1 = net[:2](X)                                  # output of the first ReLU layer
    h2 = net[:4](X)                                  # output of the second ReLU layer
    dead = (h1.max(0).values <= 0).float().mean().item()
    rank = effective_rank(h2)
    norm = torch.sqrt(sum((p ** 2).sum() for p in net.parameters())).item()
    return dead, rank, norm


def run(method, X, n_tasks, seed=0):
    set_seed(seed)
    net = make_net()
    label_gen = torch.Generator().manual_seed(123)  # same label sequence for every method
    batch_gen = torch.Generator().manual_seed(seed)
    acc, dead, rank, norm = [], [], [], []
    for t in range(n_tasks):
        y = torch.randint(0, 10, (len(X),), generator=label_gen)    # new random labels
        if t > 0 and method == "reset":
            net = make_net()
        if t > 0 and method == "shrink & perturb":
            fresh = make_net()
            with torch.no_grad():
                for p, q in zip(net.parameters(), fresh.parameters()):
                    p.mul_(0.5).add_(0.5 * q)          # shrink, then perturb with a fresh init
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(STEPS_PER_TASK):
            idx = torch.randint(0, len(X), (BATCH,), generator=batch_gen)
            loss = F.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc.append((net(X).argmax(1) == y).float().mean().item())
        d, r, n = symptoms(net, X)
        dead.append(d); rank.append(r); norm.append(n)
    return np.array(acc), np.array(dead), np.array(rank), np.array(norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="12 tasks instead of 25")
    n_tasks = 12 if parser.parse_args().quick else 25
    t0 = time.time()
    X, _ = synthetic_digits(n_per_class=500, seed=0)
    X = torch.from_numpy(X[:N_IMAGES].reshape(N_IMAGES, -1))

    methods = ["standard", "reset", "shrink & perturb"]
    res = {m: run(m, X, n_tasks) for m in methods}

    k = 5
    print(f"\nTest 20: loss of plasticity ({n_tasks} tasks of random labels, {STEPS_PER_TASK} steps each)\n")
    print(f"{'method':17s} {'acc first 5':>11s} {'acc last 5':>10s} {'dead units':>14s} {'effective rank':>16s} {'weight norm':>14s}")
    for m in methods:
        a, d, r, n = res[m]
        print(f"{m:17s} {a[:k].mean():11.3f} {a[-k:].mean():10.3f} {d[0]:6.2f} -> {d[-1]:4.2f} "
              f"{r[0]:7.0f} -> {r[-1]:4.0f} {n[0]:6.1f} -> {n[-1]:4.1f}")
    print("(Dead units: share of first-layer units that never fire. Columns with arrows run from")
    print(" the end of task 1 to the end of the last task.)")
    print("\n(Chance level is 0.10: ten classes.)")

    c = {"standard": "#C44E52", "reset": "#7F7F7F", "shrink & perturb": "#4C9A5F"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    tasks = np.arange(1, n_tasks + 1)
    for m in methods:
        a, d, r, n = res[m]
        axes[0].plot(tasks, a, color=c[m], lw=1.8, marker="o", ms=3, label=m)
        axes[1].plot(tasks, d, color=c[m], lw=1.8, label=m)
        axes[2].plot(tasks, r, color=c[m], lw=1.8, label=m)
    axes[0].axhline(0.1, color="#7F7F7F", lw=0.8, ls=":")
    axes[0].text(1, 0.12, "chance", fontsize=7.5, color="#7F7F7F", ha="left")
    axes[0].set_ylabel("training accuracy at end of task"); axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Ability to learn each new task", fontsize=10)
    axes[1].set_ylabel("share of dead units (layer 1)"); axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Dead ReLU units", fontsize=10)
    axes[2].set_ylabel("effective rank (layer 2 features)"); axes[2].set_ylim(bottom=0)
    axes[2].set_title("Variety of the features", fontsize=10)
    for ax in axes:
        ax.set_xlabel("task number"); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    out = results_dir() / "t20_plasticity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
