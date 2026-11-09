"""Test 17: EWC, protecting the weights that mattered for old tasks.

Elastic weight consolidation (Kirkpatrick et al. 2017) adds a penalty that pulls each
weight back towards the value it had after an old task, in proportion to how important
that weight was for the old task (its Fisher information F_i):

    loss = loss_new + sum over old tasks of (lambda / 2) * sum_i F_i * (theta_i - theta_old_i)^2

Part 1, permuted digits (domain-incremental, three tasks: normal pixels, then two different
pixel shuffles). Sweeping lambda shows the stability-plasticity trade-off: with lambda = 0
the oldest task is partly forgotten; as lambda grows the old task is kept, until the
network becomes so stiff that it cannot learn the new task well.

Part 2, split digits 0-4 then 5-9 with one 10-way head (class-incremental). Here EWC does
not help at all, because the weights of the new output units had zero importance for the
old task and are free to grow. This is the result van de Ven et al. (2022) found at scale.

Run:   python t17_ewc.py            (about 2 minutes)
       python t17_ewc.py --quick    (three lambda values)
Saves: results/t17_ewc.png
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

from mlguide_data import synthetic_digits, permute_pixels, set_seed, results_dir

STEPS_PER_TASK = 300
BATCH = 64
FISHER_SAMPLES = 200
LAMBDAS = [0, 1e1, 1e2, 1e3, 1e4, 1e5]


def make_net():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def fisher_diagonal(net, X, gen, n=FISHER_SAMPLES):
    """Diagonal of the Fisher information, estimated one example at a time.

    For each input we sample a label from the network's own prediction and square the
    gradient of its log-probability. Large values mark weights the old task depends on.
    """
    fisher = [torch.zeros_like(p) for p in net.parameters()]
    idx = torch.randperm(len(X), generator=gen)[:n]
    for i in idx:
        logp = F.log_softmax(net(X[i:i + 1]), dim=1)
        y = torch.multinomial(logp.exp().detach(), 1, generator=gen).item()
        net.zero_grad()
        logp[0, y].backward()
        for f, p in zip(fisher, net.parameters()):
            f += p.grad.detach() ** 2
    return [f / n for f in fisher]


def ewc_penalty(net, anchors):
    """sum over old tasks of sum_i F_i (theta_i - theta_old_i)^2 (without the lambda / 2)."""
    total = 0.0
    for fisher, old in anchors:
        for p, f, o in zip(net.parameters(), fisher, old):
            total = total + (f * (p - o) ** 2).sum()
    return total


@torch.no_grad()
def accuracy(net, X, y):
    return (net(X).argmax(1) == y).float().mean().item()


def train_sequence(tasks_train, tasks_test, lam, seed):
    """Train on the tasks in order with EWC strength lam. Returns R[i][j]."""
    set_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    anchors = []                                   # (Fisher, weights) saved after each task
    R = np.zeros((len(tasks_train), len(tasks_test)))
    for i, (X, y) in enumerate(tasks_train):
        for _ in range(STEPS_PER_TASK):
            idx = torch.randint(0, len(X), (BATCH,), generator=gen)
            loss = F.cross_entropy(net(X[idx]), y[idx])
            if lam > 0 and anchors:
                loss = loss + lam / 2 * ewc_penalty(net, anchors)
            opt.zero_grad(); loss.backward(); opt.step()
        if lam > 0:
            anchors.append((fisher_diagonal(net, X, gen),
                            [p.detach().clone() for p in net.parameters()]))
        R[i] = [accuracy(net, Xt, yt) for Xt, yt in tasks_test]
    return R


def main():
    global LAMBDAS
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="three lambda values instead of six")
    if parser.parse_args().quick:
        LAMBDAS = [0, 1e2, 1e4]
    t0 = time.time()
    Xtr, ytr = synthetic_digits(n_per_class=500, seed=0)
    Xte, yte = synthetic_digits(n_per_class=200, seed=1)
    Xtr = Xtr.reshape(len(Xtr), -1); Xte = Xte.reshape(len(Xte), -1)
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    # ---- part 1: permuted digits, three tasks
    perms = [None, 11, 12]
    def make(X, s):
        return torch.from_numpy(X if s is None else permute_pixels(X, seed=s))
    train_tasks = [(make(Xtr, s), ytr_t) for s in perms]
    test_tasks = [(make(Xte, s), yte_t) for s in perms]
    sweep = {}
    for lam in LAMBDAS:
        sweep[lam] = train_sequence(train_tasks, test_tasks, lam, seed=0)

    print("\nTest 17, part 1: EWC on permuted digits (3 tasks, test accuracy after the last task)\n")
    print(f"{'lambda':>9s} {'task 1 (oldest)':>16s} {'task 2':>8s} {'task 3 (newest)':>16s} {'average':>8s}")
    for lam in LAMBDAS:
        r = sweep[lam][-1]
        print(f"{lam:9.0f} {r[0]:16.3f} {r[1]:8.3f} {r[2]:16.3f} {r.mean():8.3f}")
    best = max([l for l in LAMBDAS if l > 0], key=lambda l: sweep[l][-1].mean())
    print(f"\nBest average at lambda = {best:.0f}. Task 1 accuracy just after learning it: {sweep[0][0, 0]:.3f}")

    # ---- part 2: class-incremental split digits, where EWC does not help
    A_tr, B_tr = ytr_t < 5, ytr_t >= 5
    A_te, B_te = yte_t < 5, yte_t >= 5
    Xtr_t, Xte_t = torch.from_numpy(Xtr), torch.from_numpy(Xte)
    split_train = [(Xtr_t[A_tr], ytr_t[A_tr]), (Xtr_t[B_tr], ytr_t[B_tr])]
    split_test = [(Xte_t[A_te], yte_t[A_te]), (Xte_t[B_te], yte_t[B_te])]
    stiff = max(LAMBDAS)
    split = {lam: train_sequence(split_train, split_test, lam, seed=0) for lam in (0, best, stiff)}
    print("\nTest 17, part 2: EWC on split digits 0-4 then 5-9 (one 10-way head)\n")
    print(f"{'lambda':>9s} {'A after A':>10s} {'A after B':>10s} {'B after B':>10s}")
    for lam, R in split.items():
        print(f"{lam:9.0f} {R[0, 0]:10.3f} {R[1, 0]:10.3f} {R[1, 1]:10.3f}")

    # ---- figure
    c = {"blue": "#3B6EA8", "orange": "#E07B39", "green": "#4C9A5F", "grey": "#7F7F7F", "purple": "#8172B2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    xs = np.arange(len(LAMBDAS))
    old = [sweep[l][-1, 0] for l in LAMBDAS]
    new = [sweep[l][-1, 2] for l in LAMBDAS]
    avg = [sweep[l][-1].mean() for l in LAMBDAS]
    ax.plot(xs, old, color=c["blue"], marker="o", lw=1.8, label="task 1 (oldest)")
    ax.plot(xs, new, color=c["orange"], marker="s", lw=1.8, label="task 3 (newest)")
    ax.plot(xs, avg, color=c["green"], marker="^", lw=1.4, ls="--", label="average of 3 tasks")
    ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"1e{int(np.log10(l))}" for l in LAMBDAS[1:]])
    ax.set_xlabel("EWC strength lambda"); ax.set_ylabel("test accuracy after task 3")
    ax.set_title("Permuted digits: the stability-plasticity trade-off", fontsize=10)
    ax.annotate("forgets", (0, old[0]), (0.25, old[0] - 0.17), fontsize=8, color=c["grey"],
                arrowprops=dict(arrowstyle="->", color=c["grey"]))
    ax.annotate("too stiff to learn", (len(LAMBDAS) - 1, new[-1]), (len(LAMBDAS) - 3.2, new[-1] - 0.12),
                fontsize=8, color=c["grey"], arrowprops=dict(arrowstyle="->", color=c["grey"]))
    ax.set_ylim(min(old + new) - 0.25, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = axes[1]
    names = ["fine-tune\n(lambda 0)", f"EWC\n(lambda 1e{int(np.log10(best))})",
             f"EWC\n(lambda 1e{int(np.log10(stiff))})"]
    a_after = [split[l][1, 0] for l in (0, best, stiff)]
    b_after = [split[l][1, 1] for l in (0, best, stiff)]
    x2 = np.arange(3)
    ax2.bar(x2 - 0.18, a_after, 0.36, color=c["blue"], label="task A (0-4)")
    ax2.bar(x2 + 0.18, b_after, 0.36, color=c["orange"], label="task B (5-9)")
    for x, a_, b_ in zip(x2, a_after, b_after):
        ax2.text(x - 0.18, a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=7)
        ax2.text(x + 0.18, b_ + 0.02, f"{b_:.2f}", ha="center", fontsize=7)
    ax2.set_xticks(x2); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylim(0, 1.25); ax2.set_yticks(np.arange(0, 1.01, 0.2)); ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylabel("test accuracy after task B")
    ax2.set_title("Split digits: EWC does not stop forgetting", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper center", ncol=2, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t17_ewc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
