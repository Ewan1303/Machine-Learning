"""Test 16: replay, fixing forgetting with a small memory.

Five tasks arrive one after another: digits 0/1, then 2/3, 4/5, 6/7 and 8/9. There is one
10-way output layer and the network is never told which task a test image comes from
(the class-incremental setting, the hardest one).

Plain fine-tuning forgets everything but the last task. Replay keeps a small memory of past
examples, chosen by reservoir sampling so every example seen so far has the same chance of
being stored, and mixes a batch from the memory into every training step.

The script tries memories of 0 (plain fine-tuning), 50, 200 and 1,000 images (out of 5,000
seen in total), three seeds each, plus joint training on all the data at once as the
upper bound. It reports:
  ACC  average accuracy over the five tasks after the last one
  BWT  backward transfer: the average change in accuracy on each old task between just
       after learning it and the end (negative means forgetting)

Run:   python t16_replay.py            (about 1 minute)
       python t16_replay.py --quick    (one seed instead of three, about a third of the time)
Saves: results/t16_replay.png
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

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
STEPS_PER_TASK = 150
BATCH = 32
MEMORY_SIZES = [0, 50, 200, 1000]
SEEDS = [0, 1, 2]


def make_net():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


class Reservoir:
    """A fixed-size memory filled by reservoir sampling (Vitter 1985).

    After n examples have streamed past, each one is in the memory with probability
    size / n, whatever order they came in, so old tasks stay represented.
    """

    def __init__(self, size, dim, gen):
        self.size, self.n_seen, self.gen = size, 0, gen
        self.X = torch.zeros(size, dim)
        self.y = torch.zeros(size, dtype=torch.long)

    def add(self, x, y):
        for xi, yi in zip(x, y):
            if self.n_seen < self.size:                 # memory not full yet: always store
                self.X[self.n_seen], self.y[self.n_seen] = xi, yi
            else:                                       # keep with probability size / (n_seen + 1)
                j = torch.randint(0, self.n_seen + 1, (1,), generator=self.gen).item()
                if j < self.size:
                    self.X[j], self.y[j] = xi, yi
            self.n_seen += 1

    def sample(self, k):
        n = min(self.n_seen, self.size)
        idx = torch.randint(0, n, (k,), generator=self.gen)
        return self.X[idx], self.y[idx]

    def __len__(self):
        return min(self.n_seen, self.size)


@torch.no_grad()
def task_accuracies(net, Xte, yte):
    pred = net(Xte).argmax(1)
    return [(pred[torch.isin(yte, torch.tensor(t))] == yte[torch.isin(yte, torch.tensor(t))]).float().mean().item()
            for t in TASKS]


def run_sequential(Xtr, ytr, Xte, yte, memory, seed):
    """Train on the tasks in order with a replay memory of the given size. Returns R[i][j]."""
    set_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    mem = Reservoir(memory, Xtr.shape[1], gen) if memory > 0 else None
    R = np.zeros((len(TASKS), len(TASKS)))
    for i, t in enumerate(TASKS):
        m = torch.isin(ytr, torch.tensor(t))
        Xt, yt = Xtr[m], ytr[m]
        for _ in range(STEPS_PER_TASK):
            idx = torch.randint(0, len(Xt), (BATCH,), generator=gen)
            xb, yb = Xt[idx], yt[idx]
            loss = F.cross_entropy(net(xb), yb)
            if mem is not None and len(mem) > 0:       # replay: add a loss on remembered examples
                xr, yr = mem.sample(BATCH)
                loss = loss + F.cross_entropy(net(xr), yr)
            opt.zero_grad(); loss.backward(); opt.step()
        if mem is not None:                            # the task's data streams into the memory
            order = torch.randperm(len(Xt), generator=gen)
            mem.add(Xt[order], yt[order])
        R[i] = task_accuracies(net, Xte, yte)
    return R


def run_joint(Xtr, ytr, Xte, yte, seed):
    """All tasks mixed together, same number of steps: the upper bound."""
    set_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_PER_TASK * len(TASKS)):
        idx = torch.randint(0, len(Xtr), (BATCH,), generator=gen)
        loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(task_accuracies(net, Xte, yte))


def acc_bwt(R):
    T = len(R)
    acc = R[-1].mean()
    bwt = np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])
    return acc, bwt


def main():
    global SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="one seed instead of three")
    if parser.parse_args().quick:
        SEEDS = [0]
    t0 = time.time()
    Xtr, ytr = synthetic_digits(n_per_class=500, seed=0)
    Xte, yte = synthetic_digits(n_per_class=200, seed=1)
    Xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1)); ytr = torch.from_numpy(ytr)
    Xte = torch.from_numpy(Xte.reshape(len(Xte), -1)); yte = torch.from_numpy(yte)

    results, matrices = {}, {}
    for M in MEMORY_SIZES:
        runs = [run_sequential(Xtr, ytr, Xte, yte, M, s) for s in SEEDS]
        matrices[M] = runs[0]
        results[M] = np.array([acc_bwt(R) for R in runs])
    joint = np.array([run_joint(Xtr, ytr, Xte, yte, s).mean() for s in SEEDS])

    print(f"\nTest 16: replay in class-incremental split digits (5 tasks, mean +/- sd over {len(SEEDS)} seed(s))\n")
    print(f"{'method':22s} {'memory':>7s} {'ACC':>15s} {'BWT':>17s}")
    for M in MEMORY_SIZES:
        name = "fine-tuning" if M == 0 else "replay (reservoir)"
        a, b = results[M][:, 0], results[M][:, 1]
        print(f"{name:22s} {M:7d} {a.mean():6.3f} +/- {a.std():.3f} {b.mean():7.3f} +/- {b.std():.3f}")
    print(f"{'joint (all data mixed)':22s} {'all':>7s} {joint.mean():6.3f} +/- {joint.std():.3f}")
    for M in (0, 200):
        print(f"\nAccuracy matrix, memory {M}, seed 0 (row = after task i, column = task j):")
        for i, row in enumerate(matrices[M]):
            print("  after task", i + 1, " ".join(f"{v:5.2f}" for v in row))

    c = {"blue": "#3B6EA8", "orange": "#E07B39", "grey": "#7F7F7F", "green": "#4C9A5F"}
    fig = plt.figure(figsize=(10.5, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0])
    xs = np.arange(len(MEMORY_SIZES))
    accs = np.array([results[M][:, 0].mean() for M in MEMORY_SIZES])
    accs_sd = np.array([results[M][:, 0].std() for M in MEMORY_SIZES])
    bwts = np.array([results[M][:, 1].mean() for M in MEMORY_SIZES])
    bwts_sd = np.array([results[M][:, 1].std() for M in MEMORY_SIZES])
    ax.errorbar(xs, accs, yerr=accs_sd, color=c["blue"], marker="o", lw=1.8, capsize=3, label="ACC (higher is better)")
    ax.errorbar(xs, bwts, yerr=bwts_sd, color=c["orange"], marker="s", lw=1.8, capsize=3, label="BWT (0 means no forgetting)")
    ax.axhline(joint.mean(), color=c["green"], ls="--", lw=1.2, label="joint training ACC")
    ax.axhline(0, color=c["grey"], lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([str(M) for M in MEMORY_SIZES])
    ax.set_xlabel("images kept in the replay memory"); ax.set_ylim(-1.05, 1.05)
    for x, a_, b_ in zip(xs, accs, bwts):
        ax.text(x, a_ + 0.07, f"{a_:.2f}", ha="center", fontsize=7.5, color=c["blue"])
        ax.text(x, b_ - 0.13, f"{b_:.2f}", ha="center", fontsize=7.5, color=c["orange"])
    ax.set_title("More memory, less forgetting", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    for k, M in enumerate((0, 200)):
        a = fig.add_subplot(gs[k + 1])
        R = matrices[M]
        mask = np.triu(np.ones_like(R), 1).astype(bool)   # tasks not yet seen
        shown = np.ma.array(R, mask=mask)
        a.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if j <= i:
                    a.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                           color="white" if R[i, j] > 0.6 else "black")
        a.set_xticks(range(5)); a.set_xticklabels([f"{p}/{q}" for p, q in TASKS], fontsize=7.5)
        a.set_yticks(range(5)); a.set_yticklabels([f"after {i + 1}" for i in range(5)], fontsize=7.5)
        a.set_xlabel("accuracy on task")
        a.set_title("Fine-tuning (memory 0)" if M == 0 else "Replay, memory 200", fontsize=10)
    fig.tight_layout()
    out = results_dir() / "t16_replay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
