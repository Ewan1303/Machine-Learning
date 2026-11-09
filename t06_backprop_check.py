"""Test 6: backpropagation by hand, and how to check it.

A two-layer network (2 inputs -> 8 tanh units -> 3 softmax outputs) is written in plain
NumPy with its backward pass coded by hand, following the equations in 1.3:

    forward:   Z1 = X W1^T + b1,  A1 = tanh(Z1),  Z2 = A1 W2^T + b2,  P = softmax(Z2)
    loss:      L = mean over examples of -log P[correct class]
    backward:  dZ2 = (P - Y) / N
               dW2 = dZ2^T A1,       db2 = sum of dZ2 over examples
               dA1 = dZ2 W2,         dZ1 = dA1 * (1 - A1^2)      (tanh' = 1 - tanh^2)
               dW1 = dZ1^T X,        db1 = sum of dZ1 over examples

The hand-written gradients are compared with two independent answers:
    1. finite differences: nudge each parameter by +-eps and measure the change in loss
    2. PyTorch autograd on the same numbers
A deliberately broken backward pass (the tanh derivative left out) shows what a failed
check looks like. Finally the hand-written gradients train the network on three spirals.

Run:
    python t06_backprop_check.py

What to look for: relative errors around 1e-9 or smaller mean the gradients are right;
errors around 1e-2 or bigger mean a bug, and the check points at the layer it is in.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlguide_data import make_spirals, plot_decision_boundary, results_dir

PARAM_NAMES = ["W1", "b1", "W2", "b2"]


def init_params(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, 1 / np.sqrt(n_in), (n_hidden, n_in)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, 1 / np.sqrt(n_hidden), (n_out, n_hidden)),
        "b2": np.zeros(n_out),
    }


def forward(p, X):
    Z1 = X @ p["W1"].T + p["b1"]
    A1 = np.tanh(Z1)
    Z2 = A1 @ p["W2"].T + p["b2"]
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)        # subtract the max: same softmax, no overflow
    P = np.exp(Z2) / np.exp(Z2).sum(axis=1, keepdims=True)
    return A1, P


def loss_fn(p, X, y):
    _, P = forward(p, X)
    return -np.mean(np.log(P[np.arange(len(y)), y]))


def backward(p, X, y, broken=False):
    """Gradients of the mean cross-entropy loss, by the chain rule, layer by layer."""
    N = len(y)
    A1, P = forward(p, X)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0                       # one-hot targets
    dZ2 = (P - Y) / N                              # softmax + cross-entropy: P - Y
    grads = {"W2": dZ2.T @ A1, "b2": dZ2.sum(axis=0)}
    dA1 = dZ2 @ p["W2"]                            # send the error back through W2
    dZ1 = dA1 if broken else dA1 * (1 - A1 ** 2)   # the bug: forgetting tanh's derivative
    grads["W1"] = dZ1.T @ X
    grads["b1"] = dZ1.sum(axis=0)
    return grads


def numerical_grads(p, X, y, eps=1e-6):
    """Central differences: (L(theta + eps) - L(theta - eps)) / (2 eps), one entry at a time."""
    grads = {}
    for name in PARAM_NAMES:
        g = np.zeros_like(p[name])
        it = np.nditer(p[name], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[name][i]
            p[name][i] = old + eps
            up = loss_fn(p, X, y)
            p[name][i] = old - eps
            down = loss_fn(p, X, y)
            p[name][i] = old
            g[i] = (up - down) / (2 * eps)
        grads[name] = g
    return grads


def autograd_grads(p, X, y):
    tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in p.items()}
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y)
    logits = torch.tanh(Xt @ tp["W1"].T + tp["b1"]) @ tp["W2"].T + tp["b2"]
    torch.nn.functional.cross_entropy(logits, yt).backward()
    return {k: v.grad.numpy() for k, v in tp.items()}


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    X, y = make_spirals(n=60, arms=3, noise=0.2, seed=0)
    X = X.astype(np.float64)
    p = init_params(2, 8, 3, rng)
    n_params = sum(v.size for v in p.values())

    bp = backward(p, X, y)
    num = numerical_grads(p, X, y)
    auto = autograd_grads(p, X, y)
    bad = backward(p, X, y, broken=True)
    print(f"Gradient check on a 2-8-3 network ({n_params} parameters), 60 examples, float64")
    print(f"{'parameter':>9s} {'backprop vs finite diff':>24s} {'backprop vs autograd':>21s} "
          f"{'broken backprop vs finite diff':>31s}")
    for name in PARAM_NAMES:
        print(f"{name:>9s} {rel_error(bp[name], num[name]):24.2e} "
              f"{rel_error(bp[name], auto[name]):21.2e} {rel_error(bad[name], num[name]):31.2e}")

    # ---- train with the hand-written gradients -------------------------------------------
    Xtr, ytr = make_spirals(n=600, arms=3, noise=0.2, seed=1)
    Xtr = Xtr.astype(np.float64)
    p = init_params(2, 64, 3, np.random.default_rng(1))
    lr, losses = 0.3, [(0, loss_fn(p, Xtr, ytr))]
    for step in range(1, 4001):
        g = backward(p, Xtr, ytr)
        for name in PARAM_NAMES:
            p[name] -= lr * g[name]                # plain gradient descent, full batch
        if step % 20 == 0:
            losses.append((step, loss_fn(p, Xtr, ytr)))
    acc = (forward(p, Xtr)[1].argmax(1) == ytr).mean()
    print(f"\nTraining a 2-64-3 NumPy network on 3 spirals with these gradients (lr {lr}, 4000 steps):")
    print(f"loss {losses[0][1]:.3f} at the start, {losses[-1][1]:.3f} at the end; "
          f"training accuracy {acc:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    s, l = zip(*losses)
    axes[0].plot(s, l, color="#3B6EA8", lw=2)
    axes[0].set_xlabel("gradient descent step", fontsize=9)
    axes[0].set_ylabel("training loss", fontsize=9)
    axes[0].set_title("Loss with hand-written gradients", fontsize=9.5)
    axes[0].grid(alpha=0.3)
    for sp in ("top", "right"):
        axes[0].spines[sp].set_visible(False)
    plot_decision_boundary(lambda G: forward(p, G.astype(np.float64))[1].argmax(1), Xtr, ytr,
                           axes[1], title=f"Decision regions, training accuracy {acc:.0%}")
    fig.tight_layout()
    out = results_dir() / "t06_backprop_check.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
