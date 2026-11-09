"""Test 4: why the starting weights matter.

The same 10-layer ReLU network (width 256) is trained on the synthetic digits five
times, changing only how the weights are initialised:

    zeros     every weight is 0
    small     weights drawn from N(0, 0.01^2)
    Xavier    N(0, 2 / (fan_in + fan_out))   (Glorot and Bengio 2010)
    He        N(0, 2 / fan_in)                (He et al. 2015), designed for ReLU
    large     weights drawn from N(0, 0.15^2)

Biases start at 0 in every case. For each start the script measures how big the
signal is at every layer before any training, then trains with plain SGD with
momentum and records the loss and the validation accuracy. A last run starts every
weight at the same constant to show that identical neurons stay identical.

Run:
    python t04_init.py

Look for: the activation size shrinking to almost nothing with "small", blowing up
with "large", and staying level with He; only the well-scaled starts learn quickly.
"""
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import results_dir, set_seed, synthetic_digits, train_val_test_split

DEPTH, WIDTH = 10, 256
EPOCHS, BATCH, LR = 12, 128, 0.01
COLOURS = {"zeros": "#7F7F7F", "small": "#8172B2", "Xavier": "#E07B39", "He": "#4C9A5F",
           "large": "#C44E52"}


def make_net():
    layers, d = [], 256
    for _ in range(DEPTH):
        layers += [nn.Linear(d, WIDTH), nn.ReLU()]
        d = WIDTH
    layers.append(nn.Linear(d, 10))
    return nn.Sequential(*layers)


def initialise(net, scheme, const=None):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            fan_in, fan_out = m.in_features, m.out_features
            with torch.no_grad():
                if scheme == "zeros":
                    m.weight.zero_()
                elif scheme == "small":
                    m.weight.normal_(0, 0.01)
                elif scheme == "Xavier":
                    m.weight.normal_(0, math.sqrt(2.0 / (fan_in + fan_out)))
                elif scheme == "He":
                    m.weight.normal_(0, math.sqrt(2.0 / fan_in))
                elif scheme == "large":
                    m.weight.normal_(0, 0.15)
                elif scheme == "constant":
                    m.weight.fill_(const / fan_in)
                m.bias.zero_()


def activation_stds(net, x):
    """Standard deviation of the output of every hidden layer for one batch."""
    stds, h = [], x
    with torch.no_grad():
        for m in net:
            h = m(h)
            if isinstance(m, nn.ReLU):
                stds.append(h.std().item())
    return stds


def accuracy(net, X, y):
    with torch.no_grad():
        return (net(X).argmax(1) == y).float().mean().item()


def train(net, Xtr, ytr, Xva, yva):
    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    losses, val_acc = [], []
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                losses.append(float("nan"))
                val_acc.append(float("nan"))
                return losses, val_acc, f"diverged in epoch {epoch + 1}"
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_acc.append(accuracy(net, Xva, yva))
    return losses, val_acc, "ok"


def main():
    t0 = time.time()
    set_seed(0)
    X, y = synthetic_digits(n_per_class=500, seed=0)
    (Xtr, ytr), (Xva, yva), _ = train_val_test_split(X.reshape(len(X), -1), y, val=0.2, test=0.0)
    Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))

    schemes = ["zeros", "small", "Xavier", "He", "large"]
    results = {}
    print(f"{'start':8s} {'std L1':>9s} {'std L5':>9s} {'std L10':>9s} {'first loss':>11s} "
          f"{'last loss':>10s} {'val acc':>8s}  note")
    for s in schemes:
        set_seed(1)
        net = make_net()
        initialise(net, s)
        stds = activation_stds(net, Xtr[:512])
        first_loss = F.cross_entropy(net(Xtr[:512]), ytr[:512]).item()
        losses, val_acc, note = train(net, Xtr, ytr, Xva, yva)
        results[s] = dict(stds=stds, losses=losses, val_acc=val_acc, note=note)
        last = np.nanmean(losses[-20:]) if note == "ok" else float("nan")
        acc = val_acc[-1] if note == "ok" else float("nan")
        print(f"{s:8s} {stds[0]:9.2e} {stds[4]:9.2e} {stds[9]:9.2e} {first_loss:11.3g} "
              f"{last:10.3f} {acc:8.1%}  {note}")

    # Identical starting weights stay identical: count distinct neurons after training
    set_seed(1)
    net = make_net()
    initialise(net, "constant", const=1.0)
    losses, val_acc, note = train(net, Xtr, ytr, Xva, yva)
    W1 = net[0].weight.detach()
    distinct = torch.unique(torch.round(W1 * 1e5), dim=0).shape[0]
    print(f"\nConstant start: after training, layer 1 has {distinct} distinct neuron(s) out of "
          f"{W1.shape[0]}; val acc {val_acc[-1]:.1%}")
    print(f"ln(10) = {math.log(10):.3f} is the loss of a network that guesses uniformly at random")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax = axes[0]
    for s in schemes:
        st = np.array(results[s]["stds"])
        st = np.where(st > 0, st, np.nan)
        ax.semilogy(range(1, DEPTH + 1), st, "o-", ms=3, color=COLOURS[s], label=s)
    ax.semilogy([1, DEPTH], [1e-12, 1e-12], alpha=0)  # keep axis range when zeros are missing
    ax.set_xlabel("hidden layer", fontsize=9)
    ax.set_ylabel("std of activations at the start", fontsize=9)
    ax.set_title("Signal size through the network", fontsize=9.5)
    ax.text(1.2, 3e-12, "zeros: exactly 0 at every layer", fontsize=7.5, color=COLOURS["zeros"])
    ax = axes[1]
    for s in schemes:
        va = results[s]["val_acc"]
        ax.plot(range(1, len(va) + 1), [v * 100 for v in va], "o-", ms=3, color=COLOURS[s], label=s)
    ax.set_xlabel("epoch", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("validation accuracy (%)", fontsize=9)
    ax.set_title("Learning", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    ax.text(4.0, 14, "zeros and small: stuck at chance", fontsize=7.5, color=COLOURS["zeros"])
    if results["large"]["note"] != "ok":
        ax.text(4.0, 22, f"large: {results['large']['note']} (loss NaN)", fontsize=7.5,
                color=COLOURS["large"])
    for a in axes:
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t04_init.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
