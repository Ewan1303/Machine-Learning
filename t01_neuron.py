"""Test 1: a single neuron and its decision line.

A neuron with two inputs computes z = w1*x1 + w2*x2 + b and squashes it with a
sigmoid, a = 1 / (1 + exp(-z)). The points where z = 0 form a straight line: the
neuron's decision boundary. This script moves the weights and the bias by hand to
show what each one does to that line, then trains a single neuron (logistic
regression) on two clusters of points and prints what it learned.

Run:
    python t01_neuron.py

What to look for:
    - w sets the direction of the line: the vector w points straight across it,
      towards the side where the output is above 0.5.
    - b slides the line without turning it. The line sits |b| / ||w|| from the origin.
    - Multiplying w and b by the same number keeps the line where it is but makes
      the output switch from 0 to 1 more sharply (the switch is 4 / ||w|| wide).
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def neuron(X, w, b):
    """The whole neuron: a weighted sum plus a bias, then a sigmoid."""
    return sigmoid(X @ w + b)


def draw_neuron(ax, w, b, title, lim=3.0):
    """Shade the neuron's output over the plane and draw its line z = 0 and the vector w."""
    g = np.linspace(-lim, lim, 300)
    gx, gy = np.meshgrid(g, g)
    out = neuron(np.stack([gx.ravel(), gy.ravel()], 1), w, b).reshape(gx.shape)
    ax.imshow(out, extent=(-lim, lim, -lim, lim), origin="lower", cmap="coolwarm",
              vmin=0, vmax=1, alpha=0.8)
    # the line w1*x1 + w2*x2 + b = 0, drawn through its closest point to the origin
    n = w / np.linalg.norm(w)
    p0 = -b / np.linalg.norm(w) * n              # closest point on the line to the origin
    d = np.array([-n[1], n[0]])                    # direction along the line
    seg = np.array([p0 - 10 * d, p0 + 10 * d])
    ax.plot(seg[:, 0], seg[:, 1], color="k", lw=1.6)
    ax.annotate("", xy=p0 + 0.9 * n, xytext=p0,
                arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
    ax.text(*(p0 + 1.0 * n + np.array([0.1, 0.1])), "w", fontsize=10, weight="bold")
    ax.plot(0, 0, marker="+", color="k", ms=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)


def main():
    t0 = time.time()
    set_seed(0)

    # ---- Part 1: move w and b by hand ------------------------------------------
    settings = [
        ("start: w = (1, 1), b = 0", np.array([1.0, 1.0]), 0.0),
        ("turn w: w = (1, -0.5), b = 0", np.array([1.0, -0.5]), 0.0),
        ("shift: w = (1, 1), b = -2", np.array([1.0, 1.0]), -2.0),
        ("sharpen: w = (4, 4), b = -8", np.array([4.0, 4.0]), -8.0),
    ]
    print("Part 1: what w and b do to the decision line")
    print(f"{'setting':32s} {'distance |b|/||w||':>20s} {'switch width 4/||w||':>22s}")
    for name, w, b in settings:
        norm = np.linalg.norm(w)
        print(f"{name:32s} {abs(b) / norm:20.3f} {4 / norm:22.3f}")

    # ---- Part 2: train one neuron (logistic regression) -------------------------
    X, y = make_blobs(n=400, centers=((0.0, 0.5), (2.0, 2.5)), std=0.6, seed=1)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).float()
    model = nn.Linear(2, 1)                       # one neuron: 2 weights and 1 bias
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()              # sigmoid + cross-entropy in one stable step
    for step in range(1500):
        opt.zero_grad()
        loss = loss_fn(model(Xt).squeeze(1), yt)
        loss.backward()
        opt.step()
    w = model.weight.detach().numpy().ravel()
    b = float(model.bias.detach())
    acc = ((model(Xt).squeeze(1) > 0).float() == yt).float().mean().item()
    print("\nPart 2: one neuron trained on two clusters")
    print(f"learned w = ({w[0]:.3f}, {w[1]:.3f}), b = {b:.3f}")
    print(f"line sits {abs(b) / np.linalg.norm(w):.3f} from the origin; "
          f"switch width 4/||w|| = {4 / np.linalg.norm(w):.3f}")
    print(f"final loss {loss.item():.4f}, training accuracy {acc:.3f}")

    # ---- Figure ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    for ax, (name, wv, bv) in zip(axes.ravel()[:4], settings):
        draw_neuron(ax, wv, bv, name)
    ax = axes[1, 1]
    draw_neuron(ax, w, b, "trained on the data", lim=3.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8, edgecolors="k", linewidths=0.2)
    ax = axes[1, 2]
    s = np.linspace(-3, 3, 300)
    for scale, col in [(1, BLUE), (4, ORANGE)]:
        ax.plot(s, sigmoid(scale * np.sqrt(2) * s), color=col, lw=2,
                label=f"||w|| = {scale * np.sqrt(2):.1f}")
    ax.set_xlabel("signed distance from the line", fontsize=9)
    ax.set_ylabel("neuron output", fontsize=9)
    ax.set_title("bigger weights, sharper switch", fontsize=9.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8.5, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t01_neuron.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
