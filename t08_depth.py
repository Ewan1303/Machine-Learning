"""Test 8: what depth does, and why residual connections were invented.

Three kinds of network learn the two-spirals problem at depths from 2 to 48 hidden
layers (width 32 throughout):

    plain sigmoid   Linear -> sigmoid, repeated (Xavier start)
    plain ReLU      Linear -> ReLU, repeated (He start)
    residual ReLU   blocks that compute h + W2 ReLU(W1 h), so each block starts as the
                    identity and the gradient has a direct path back (He et al. 2016)

First the script measures the gradient that reaches each layer before training, and
how that changes from 8 to 96 layers. Then it trains every network with the same
budget (full-batch Adam) and reports the training accuracy and how many steps it took
to reach 99%.

Run:
    python t08_depth.py            (about a minute)
    python t08_depth.py --quick

Look for: in the plain sigmoid network the gradient shrinks by orders of magnitude on
its way to the first layers, so deep sigmoid networks barely learn. With ReLU and He
initialisation the gradient still fades with depth, but slowly; in the residual
network it arrives at every layer at the same size.
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

from mlguide_data import make_spirals, results_dir, set_seed

WIDTH = 32
COLOURS = {"plain sigmoid": "#8172B2", "plain ReLU": "#E07B39", "residual ReLU": "#4C9A5F"}


class Plain(nn.Module):
    def __init__(self, depth, act):
        super().__init__()
        self.inp = nn.Linear(2, WIDTH)
        self.hidden = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(depth))
        self.out = nn.Linear(WIDTH, 2)
        self.act = torch.sigmoid if act == "sigmoid" else torch.relu
        for m in [self.inp, *self.hidden]:
            if act == "sigmoid":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x, keep=False):
        h = self.act(self.inp(x))
        self.trace = []                    # hidden states, kept only for the gradient check
        for layer in self.hidden:
            h = self.act(layer(h))
            if keep:
                h.retain_grad()
                self.trace.append(h)
        return self.out(h)

    def positions(self):
        return list(range(1, len(self.hidden) + 1))


class Residual(nn.Module):
    """depth hidden Linear layers arranged as depth/2 blocks of h + W2 relu(W1 h)."""

    def __init__(self, depth):
        super().__init__()
        self.inp = nn.Linear(2, WIDTH)
        self.w1 = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(depth // 2))
        self.w2 = nn.ModuleList(nn.Linear(WIDTH, WIDTH) for _ in range(depth // 2))
        self.out = nn.Linear(WIDTH, 2)
        nn.init.kaiming_normal_(self.inp.weight, nonlinearity="relu")
        for a, b in zip(self.w1, self.w2):
            nn.init.kaiming_normal_(a.weight, nonlinearity="relu")
            nn.init.zeros_(a.bias)
            nn.init.zeros_(b.weight)       # each block starts as the identity map
            nn.init.zeros_(b.bias)

    def forward(self, x, keep=False):
        h = torch.relu(self.inp(x))
        self.trace = []
        for a, b in zip(self.w1, self.w2):
            h = h + b(torch.relu(a(h)))    # the skip connection: add the input back
            if keep:
                h.retain_grad()
                self.trace.append(h)
        return self.out(h)

    def positions(self):
        return list(range(2, 2 * len(self.w1) + 1, 2))   # each block holds two layers


def build(kind, depth):
    if kind == "plain sigmoid":
        return Plain(depth, "sigmoid")
    if kind == "plain ReLU":
        return Plain(depth, "relu")
    return Residual(depth)


def grad_norms(net, X, y):
    """Size of dL/dh for the hidden state h after each layer (or block), before training."""
    loss = F.cross_entropy(net(X, keep=True), y)
    net.zero_grad()
    loss.backward()
    return net.positions(), [h.grad.norm().item() for h in net.trace]


def train(net, X, y, steps):
    """Adam, full batch. Returns final loss, final accuracy and the first step at 99% accuracy."""
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    first99 = None
    for step in range(1, steps + 1):
        out = net(X)
        loss = F.cross_entropy(out, y)
        if first99 is None and (out.argmax(1) == y).float().mean().item() >= 0.99:
            first99 = step
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = net(X)
        return (F.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item(),
                first99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    X, y = make_spirals(n=400, noise=0.2, seed=0)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    kinds = list(COLOURS)

    # 1. Gradient reaching each layer of a 32-layer network, before any training
    norms = {}
    for k in kinds:
        set_seed(0)
        norms[k] = grad_norms(build(k, 32), X, y)
    print("Gradient reaching the hidden state after each layer (norm of dL/dh), 32 layers, "
          "before training:")
    print(f"{'network':15s} {'near input':>11s} {'middle':>10s} {'near output':>12s} "
          f"{'input/output':>13s}")
    for k in kinds:
        pos, g = norms[k]
        print(f"{k:15s} {g[0]:11.2e} {g[len(g) // 2]:10.2e} {g[-1]:12.2e} {g[0] / g[-1]:13.1e}")
    print("\nRatio of gradient size near the input to near the output, by depth:")
    print(f"{'network':15s} " + " ".join(f"{d:>9d}" for d in (8, 32, 96)))
    for k in kinds:
        ratios = []
        for d in (8, 32, 96):
            set_seed(0)
            _, g = grad_norms(build(k, d), X, y)
            ratios.append(g[0] / g[-1])
        print(f"{k:15s} " + " ".join(f"{r:9.1e}" for r in ratios))

    # 2. Same training budget at every depth
    depths = [2, 8, 32] if args.quick else [2, 8, 16, 32, 48]
    steps = 500 if args.quick else 800
    res = {k: [] for k in kinds}
    print(f"\nAfter {steps} full-batch Adam steps (learning rate 0.002):")
    print(f"{'depth':>5s} " + " ".join(f"{k:>24s}" for k in kinds))
    print(f"{'':5s} " + " ".join(f"{'acc  (step at 99%)':>24s}" for _ in kinds))
    for d in depths:
        line = f"{d:5d} "
        for k in kinds:
            set_seed(0)
            loss, a, f99 = train(build(k, d), X, y, steps)
            res[k].append((loss, a, f99))
            line += f"{a:13.1%} {('(' + str(f99) + ')') if f99 else '(never)':>10s}"
        print(line, flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
    ax = axes[0]
    for k in kinds:
        pos, g = norms[k]
        ax.semilogy(pos, g, "o-", ms=2, color=COLOURS[k], label=k)
    ax.set_xlabel("layer (1 = nearest input)", fontsize=8)
    ax.set_ylabel("gradient size at the start", fontsize=8)
    ax.set_title("Gradient per layer, depth 32", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axes[1]
    for k in kinds:
        acc = [r[1] * 100 for r in res[k]]
        ax.plot(depths, acc, "o-", ms=3, color=COLOURS[k], label=k)
    ax.set_xlabel("hidden layers", fontsize=8)
    ax.set_ylabel("training accuracy (%)", fontsize=8)
    ax.set_title(f"Accuracy after {steps} steps", fontsize=8.5)
    ax.set_ylim(40, 104)
    ax.text(depths[-1], 96, "ReLU and residual\nboth reach 100%", fontsize=6.5, ha="right", va="top",
            color="#555555")
    ax = axes[2]
    for k in kinds[1:]:
        st = [r[2] if r[2] else np.nan for r in res[k]]
        ax.plot(depths, st, "o-", ms=3, color=COLOURS[k], label=k)
    ax.set_ylim(0, None)
    ax.set_xlabel("hidden layers", fontsize=8)
    ax.set_ylabel("steps to reach 99% accuracy", fontsize=8)
    ax.set_title("Training speed (sigmoid: never)", fontsize=8.5)
    for a in axes:
        a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t08_depth.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
