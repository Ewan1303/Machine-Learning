"""Test 15: watching catastrophic forgetting happen.

A small network learns task A, then task B, and we measure how well it still does task A
while it learns B. Two kinds of task sequence are tried:

  1. Split digits (class-incremental): task A is the digits 0-4, task B is 5-9, one
     10-way output layer shared by both.
  2. Permuted digits (domain-incremental): task A is the normal images, task B is the same
     digits with their pixels shuffled by one fixed permutation. Same 10 labels.

For each, a "joint" run trains on A and B mixed together for the same number of steps.
That shows the network is big enough to do both tasks at once, so whatever is lost in the
sequential run is caused by the order of the data, not by a lack of capacity.

Run:   python t15_forgetting.py
Look:  the accuracy on task A in the first panels as soon as task B starts.
Saves: results/t15_forgetting.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import synthetic_digits, permute_pixels, set_seed, results_dir

STEPS_PER_TASK = 300   # optimiser steps on each task
BATCH = 64
EVAL_EVERY = 10        # steps between accuracy checks


def make_net():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


@torch.no_grad()
def accuracy(net, X, y, allowed=None):
    """Accuracy on (X, y). With `allowed` set, only those classes may be predicted."""
    logits = net(X)
    if allowed is not None:
        mask = torch.full((10,), float("-inf"))
        mask[allowed] = 0.0
        logits = logits + mask          # rule out every class outside the task
    return (logits.argmax(1) == y).float().mean().item()


def train(net, opt, X, y, steps, gen, evals, curve):
    """Run `steps` minibatch updates on (X, y), recording every evaluation in `curve`."""
    for s in range(steps):
        idx = torch.randint(0, len(X), (BATCH,), generator=gen)
        loss = F.cross_entropy(net(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % EVAL_EVERY == 0:
            curve.append([f(net) for f in evals])


def main():
    t0 = time.time()
    set_seed(0)
    Xtr, ytr = synthetic_digits(n_per_class=500, seed=0)    # 5,000 training images
    Xte, yte = synthetic_digits(n_per_class=200, seed=1)    # 2,000 test images
    Xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1)); ytr = torch.from_numpy(ytr)
    Xte = torch.from_numpy(Xte.reshape(len(Xte), -1)); yte = torch.from_numpy(yte)

    # ---- scenario 1: split digits, 0-4 then 5-9, one shared 10-way head
    inA, inB = ytr < 5, ytr >= 5
    teA, teB = yte < 5, yte >= 5
    evals_split = [
        lambda n: accuracy(n, Xte[teA], yte[teA]),                       # A, any of 10 classes
        lambda n: accuracy(n, Xte[teA], yte[teA], allowed=[0, 1, 2, 3, 4]),  # A, told it is task A
        lambda n: accuracy(n, Xte[teB], yte[teB]),                       # B
    ]
    set_seed(1)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(1)
    split_curve = []
    train(net, opt, Xtr[inA], ytr[inA], STEPS_PER_TASK, gen, evals_split, split_curve)
    split_after_A = split_curve[-1]
    train(net, opt, Xtr[inB], ytr[inB], STEPS_PER_TASK, gen, evals_split, split_curve)
    split_after_B = split_curve[-1]

    set_seed(1)                                # joint: both halves mixed, same total steps
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    joint_curve = []
    train(net, opt, Xtr, ytr, 2 * STEPS_PER_TASK, torch.Generator().manual_seed(1), evals_split, joint_curve)
    split_joint = joint_curve[-1]

    # ---- scenario 2: permuted digits, normal images then pixel-shuffled images
    XtrP = torch.from_numpy(permute_pixels(Xtr.numpy(), seed=7))
    XteP = torch.from_numpy(permute_pixels(Xte.numpy(), seed=7))
    evals_perm = [lambda n: accuracy(n, Xte, yte), lambda n: accuracy(n, XteP, yte)]
    set_seed(2)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(2)
    perm_curve = []
    train(net, opt, Xtr, ytr, STEPS_PER_TASK, gen, evals_perm, perm_curve)
    perm_after_A = perm_curve[-1]
    train(net, opt, XtrP, ytr, STEPS_PER_TASK, gen, evals_perm, perm_curve)
    perm_after_B = perm_curve[-1]

    set_seed(2)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xmix, ymix = torch.cat([Xtr, XtrP]), torch.cat([ytr, ytr])
    joint_p = []
    train(net, opt, Xmix, ymix, 2 * STEPS_PER_TASK, torch.Generator().manual_seed(2), evals_perm, joint_p)
    perm_joint = joint_p[-1]

    # ---- print the numbers
    print("\nTest 15: catastrophic forgetting (test-set accuracy)\n")
    print(f"{'scenario':38s} {'A after A':>9s} {'A after B':>9s} {'B after B':>9s}")
    print(f"{'split digits, 10-way head':38s} {split_after_A[0]:9.3f} {split_after_B[0]:9.3f} {split_after_B[2]:9.3f}")
    print(f"{'split digits, told the task (0-4 only)':38s} {split_after_A[1]:9.3f} {split_after_B[1]:9.3f} {'':>9s}")
    print(f"{'permuted digits':38s} {perm_after_A[0]:9.3f} {perm_after_B[0]:9.3f} {perm_after_B[1]:9.3f}")
    print("\nJoint training on A and B mixed, same number of steps:")
    print(f"  split digits:    A {split_joint[0]:.3f}   B {split_joint[2]:.3f}")
    print(f"  permuted digits: A {perm_joint[0]:.3f}   B {perm_joint[1]:.3f}")

    # ---- plot
    c = {"blue": "#3B6EA8", "orange": "#E07B39", "green": "#4C9A5F", "grey": "#7F7F7F"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    steps = np.arange(1, len(split_curve) + 1) * EVAL_EVERY
    sc = np.array(split_curve)
    ax = axes[0]
    ax.plot(steps, sc[:, 0], color=c["blue"], lw=1.8, label="A, 10-way")
    ax.plot(steps, sc[:, 1], color=c["blue"], lw=1.4, ls="--", label="A, told the task")
    ax.plot(steps, sc[:, 2], color=c["orange"], lw=1.8, label="B (5-9)")
    ax.set_title("Split digits: 0-4, then 5-9", fontsize=10)
    pc = np.array(perm_curve)
    ax2 = axes[1]
    ax2.plot(steps, pc[:, 0], color=c["blue"], lw=1.8, label="A (normal)")
    ax2.plot(steps, pc[:, 1], color=c["orange"], lw=1.8, label="B (permuted)")
    ax2.set_title("Permuted digits: normal, then shuffled", fontsize=10)
    for a in (ax, ax2):
        a.axvline(STEPS_PER_TASK, color=c["grey"], lw=1, ls=":")
        a.text(STEPS_PER_TASK / 2, 1.07, "training on A", fontsize=8, color=c["grey"], ha="center")
        a.text(STEPS_PER_TASK * 1.5, 1.07, "training on B", fontsize=8, color=c["grey"], ha="center")
        a.set_xlabel("training step"); a.set_ylabel("test accuracy")
        a.set_ylim(-0.02, 1.14); a.set_yticks(np.arange(0, 1.01, 0.2)); a.grid(alpha=0.3)
        a.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.06, 0.45), framealpha=0.9)
        a.spines[["top", "right"]].set_visible(False)
    ax3 = axes[2]
    labels = ["split\nsequential", "split\njoint", "permuted\nsequential", "permuted\njoint"]
    accA = [split_after_B[0], split_joint[0], perm_after_B[0], perm_joint[0]]
    accB = [split_after_B[2], split_joint[2], perm_after_B[1], perm_joint[1]]
    xs = np.arange(4)
    ax3.bar(xs - 0.18, accA, 0.36, color=c["blue"], label="task A")
    ax3.bar(xs + 0.18, accB, 0.36, color=c["orange"], label="task B")
    for x, a_, b_ in zip(xs, accA, accB):
        ax3.text(x - 0.18, a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=7)
        ax3.text(x + 0.18, b_ + 0.02, f"{b_:.2f}", ha="center", fontsize=7)
    ax3.set_xticks(xs); ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylim(0, 1.25); ax3.set_yticks(np.arange(0, 1.01, 0.2))
    ax3.set_ylabel("final test accuracy"); ax3.grid(alpha=0.3, axis="y")
    ax3.set_title("At the end of training", fontsize=10)
    ax3.legend(fontsize=7.5, loc="upper center", ncol=2, frameon=False)
    ax3.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t15_forgetting.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
