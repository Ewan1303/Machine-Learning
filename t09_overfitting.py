"""Test 9: overfitting, and four ways to fight it.

A network with three hidden layers of 128 (about 33,700 parameters) is trained on only
80 points from the two-moons problem, and 20% of those training labels are wrong
(flipped on purpose, as happens with real labelling mistakes). The network has far more
freedom than the data can pin down, so it can memorise the mistakes. The script compares:

    none            plain training for 2,000 full-batch Adam steps
    L2 penalty      adds lambda * (sum of squared weights) to the loss, lambda = 0.01
    dropout         randomly switches off 30% of hidden units at each step
    early stopping  keeps the weights from the step with the lowest validation loss
    more data       no regularisation, but 800 training points (still 20% mislabelled)

Accuracy is measured on 2,000 fresh, correctly labelled test points. The 500 validation
points are used only to choose the early-stopping step.

Run:
    python t09_overfitting.py

Look for: the unregularised network reaches 100% training accuracy, which means it has
learned the wrong labels too, and its boundary grows islands around them; its
validation loss falls and then rises. The L2 penalty and early stopping refuse to fit
the mislabelled points (training accuracy near 80%) and score about 10 points higher
on the test set.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import count_params, make_moons, plot_decision_boundary, results_dir, set_seed

STEPS, LR, HIDDEN = 2000, 1e-3, 128
FLIP = 0.2   # fraction of training labels made wrong


def big_net(p_drop=0.0):
    layers, d = [], 2
    for _ in range(3):
        layers += [nn.Linear(d, HIDDEN), nn.ReLU()]
        if p_drop:
            layers.append(nn.Dropout(p_drop))
        d = HIDDEN
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers)


def flip_labels(y, frac, seed):
    """Make a fraction of the labels wrong, chosen at random."""
    rng = np.random.default_rng(seed)
    y = y.copy()
    idx = rng.choice(len(y), int(frac * len(y)), replace=False)
    y[idx] = 1 - y[idx]
    return y


def l2(net):
    """Sum of squared weights (biases are usually left out of the penalty)."""
    return sum((m.weight ** 2).sum() for m in net if isinstance(m, nn.Linear))


def evaluate(net, X, y):
    net.eval()                      # switches dropout off
    with torch.no_grad():
        out = net(X)
        loss, acc = F.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()
    net.train()
    return loss, acc


def train(net, X, y, Xva, yva, lam=0.0, early_stop=False):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    curve, best = [], (float("inf"), 0, None)
    for step in range(1, STEPS + 1):
        loss = F.cross_entropy(net(X), y)
        if lam:
            loss = loss + lam * l2(net)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0:
            tr_loss, _ = evaluate(net, X, y)
            va_loss, _ = evaluate(net, Xva, yva)
            curve.append((step, tr_loss, va_loss))
            if va_loss < best[0]:
                best = (va_loss, step, {k: v.clone() for k, v in net.state_dict().items()})
    if early_stop:
        net.load_state_dict(best[2])  # go back to the weights with the lowest validation loss
    return curve, best[1]


def main():
    t0 = time.time()
    Xtr, ytr = make_moons(n=80, noise=0.25, seed=1)
    ytr = flip_labels(ytr, FLIP, seed=5)
    Xbig, ybig = make_moons(n=800, noise=0.25, seed=4)
    ybig = flip_labels(ybig, FLIP, seed=6)
    Xva, yva = make_moons(n=500, noise=0.25, seed=2)
    Xte, yte = make_moons(n=2000, noise=0.25, seed=3)
    Xtr, ytr, Xbig, ybig, Xva, yva, Xte, yte = map(
        torch.from_numpy, (Xtr, ytr, Xbig, ybig, Xva, yva, Xte, yte))

    runs = [("none", dict()), ("L2 penalty", dict(lam=1e-2)), ("dropout", dict(p=0.3)),
            ("early stopping", dict(early=True)), ("more data", dict(big=True))]
    nets, curves = {}, {}
    print(f"{'method':15s} {'train acc':>10s} {'test acc':>9s}  note")
    for name, cfg in runs:
        set_seed(0)
        net = big_net(cfg.get("p", 0.0))
        X, y = (Xbig, ybig) if cfg.get("big") else (Xtr, ytr)
        curve, best_step = train(net, X, y, Xva, yva, lam=cfg.get("lam", 0.0),
                                 early_stop=cfg.get("early", False))
        _, tr_acc = evaluate(net, X, y)
        _, te_acc = evaluate(net, Xte, yte)
        note = ""
        if cfg.get("early"):
            note = f"stopped at step {best_step} of {STEPS}"
        if cfg.get("big"):
            note = "800 training points"
        print(f"{name:15s} {tr_acc:10.1%} {te_acc:9.1%}  {note}")
        nets[name], curves[name] = net, (curve, best_step)
    print(f"\nParameters in each network: {count_params(big_net()):,}; training points: 80, "
          f"of which {int(FLIP * 80)} have the wrong label")
    print("Training accuracy is measured against the noisy labels, so 100% means the wrong "
          "labels were memorised.")

    fig = plt.figure(figsize=(6.4, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    for k, name in enumerate(["none", "L2 penalty", "early stopping", "more data"]):
        a = fig.add_subplot(gs[0, k])
        net = nets[name]
        net.eval()
        X, y = (Xbig, ybig) if name == "more data" else (Xtr, ytr)
        with torch.no_grad():
            plot_decision_boundary(lambda g: net(torch.from_numpy(g)).argmax(1).numpy(),
                                   X.numpy(), y.numpy(), a, title=name)
    ax = fig.add_subplot(gs[1, :])
    curve, best_step = curves["none"]
    c = np.array(curve)
    ax.plot(c[:, 0], c[:, 1], color="#3B6EA8", label="training loss (no regularisation)")
    ax.plot(c[:, 0], c[:, 2], color="#E07B39", label="validation loss (no regularisation)")
    ax.axvline(best_step, color="#7F7F7F", ls="--", lw=1)
    ax.text(best_step + 30, c[:, 2].max() * 0.8,
            f"lowest validation loss: early stopping\nwould keep the weights from step {best_step}",
            fontsize=7.5, color="#555555")
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("cross-entropy loss", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t09_overfitting.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
