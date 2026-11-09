"""Test 2: why biases matter.

Two experiments, each run with and without bias terms.

Part A, classification. Two clusters sit on the same diagonal, one near (1.5, 1.5)
and one near (3.5, 3.5). A single neuron with a bias can put its line between them.
Without a bias the line has to pass through the origin, and every line through the
origin has both clusters on the same side.

Part B, regression. A ReLU network (1 -> 32 -> 32 -> 1) is asked to fit a bump,
y = exp(-2 (x - 2)^2), for x between 0 and 4. With biases it fits easily. Without
biases every hidden unit computes ReLU(w x), which bends only at x = 0, so on the
whole range 0 < x < 4 the network can only draw a straight line through the origin.
The script also checks two side effects of having no bias: f(0) is exactly 0, and
f(2x) = 2 f(x) for every x > 0.

Run:
    python t02_bias.py

What to look for: accuracy near 50% and a straight-line "fit" for the bias-free models.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mlguide_data import make_blobs, results_dir, set_seed

BLUE, ORANGE, GREEN, RED, GREY = "#3B6EA8", "#E07B39", "#4C9A5F", "#C44E52", "#7F7F7F"


def train(model, X, y, loss_fn, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    return loss.item()


def mlp(bias):
    return nn.Sequential(nn.Linear(1, 32, bias=bias), nn.ReLU(),
                         nn.Linear(32, 32, bias=bias), nn.ReLU(),
                         nn.Linear(32, 1, bias=bias))


def main():
    t0 = time.time()

    # ---- Part A: one neuron, with and without a bias ----------------------------
    X, y = make_blobs(n=400, centers=((1.5, 1.5), (3.5, 3.5)), std=0.5, seed=0)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float().unsqueeze(1)
    results_a = {}
    for bias in (True, False):
        set_seed(0)
        neuron = nn.Linear(2, 1, bias=bias)
        loss = train(neuron, Xt, yt, nn.BCEWithLogitsLoss(), steps=2000, lr=0.05)
        acc = ((neuron(Xt) > 0).float() == yt).float().mean().item()
        w = neuron.weight.detach().numpy().ravel()
        b = float(neuron.bias.detach()) if bias else 0.0
        results_a[bias] = (loss, acc, w, b)

    print("Part A: one neuron separating two clusters on the same diagonal")
    print(f"{'model':18s} {'final loss':>11s} {'accuracy':>9s}   learned line")
    for bias in (True, False):
        loss, acc, w, b = results_a[bias]
        name = "with bias" if bias else "without bias"
        print(f"{name:18s} {loss:11.4f} {acc:9.3f}   {w[0]:.2f} x1 + {w[1]:.2f} x2 + {b:.2f} = 0")

    # ---- Part B: a ReLU network fitting a bump ---------------------------------------
    xs = torch.linspace(0, 4, 200).unsqueeze(1)
    ys = torch.exp(-2 * (xs - 2) ** 2)
    nets, results_b = {}, {}
    for bias in (True, False):
        set_seed(0)
        net = mlp(bias)
        mse = train(net, xs, ys, nn.MSELoss(), steps=3000, lr=0.01)
        nets[bias] = net
        with torch.no_grad():
            f0 = net(torch.zeros(1, 1)).item()
            x_probe = torch.tensor([[0.7], [1.3], [2.9]])
            ratio = (net(2 * x_probe) / net(x_probe)).squeeze().tolist()
        results_b[bias] = (mse, f0, ratio)

    print("\nPart B: a 1-32-32-1 ReLU network fitting y = exp(-2 (x - 2)^2) on [0, 4]")
    print(f"{'model':18s} {'final MSE':>10s} {'f(0)':>9s}   f(2x)/f(x) at x = 0.7, 1.3, 2.9")
    for bias in (True, False):
        mse, f0, ratio = results_b[bias]
        name = "with biases" if bias else "without biases"
        print(f"{name:18s} {mse:10.5f} {f0:9.4f}   " + ", ".join(f"{r:.3f}" for r in ratio))
    print(f"(for reference, the best straight line through the origin has MSE "
          f"{float(((ys - (xs * (xs * ys).sum() / (xs * xs).sum())) ** 2).mean()):.5f})")

    # ---- Figure --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=9, edgecolors="k", linewidths=0.2)
    gx = np.linspace(-0.5, 5.5, 50)
    for bias, col, ls in [(True, GREEN, "-"), (False, RED, "--")]:
        _, acc, w, b = results_a[bias]
        gy = -(w[0] * gx + b) / w[1]
        lab = f"{'with' if bias else 'without'} bias ({acc:.0%} correct)"
        ax.plot(gx, gy, color=col, ls=ls, lw=2, label=lab)
    ax.plot(0, 0, "k+", ms=10)
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 5.5)
    ax.set_title("A: the best line with and without a bias", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax = axes[1]
    xx = torch.linspace(-1, 5, 400).unsqueeze(1)
    ax.plot(xs.squeeze(), ys.squeeze(), color=GREY, lw=5, alpha=0.35, label="target bump")
    with torch.no_grad():
        ax.plot(xx.squeeze(), nets[True](xx).squeeze(), color=GREEN, lw=2, label="ReLU net with biases")
        ax.plot(xx.squeeze(), nets[False](xx).squeeze(), color=RED, lw=2, ls="--",
                label="ReLU net without biases")
    ax.axvspan(0, 4, color="#EEEEEE", zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("output", fontsize=9)
    ax.set_title("B: fitting a bump (shaded: training range)", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = results_dir() / "t02_bias.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
