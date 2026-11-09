"""Task 11 solution: a continual-learning mini-project over five tasks.

    python task_11_continual.py            # all methods, about a minute
    python task_11_continual.py --lambdas  # also sweep the EWC strength

The digits arrive in five tasks, two classes at a time: (0,1), (2,3), (4,5), (6,7), (8,9).
One network with a single 10-way output head learns them in order. Methods compared:
  fine-tune      plain training on each task in turn (the lower bound)
  EWC            elastic weight consolidation (Kirkpatrick et al. 2017)
  replay         a 200-example reservoir buffer mixed into every batch
  replay + EWC   both
  joint          all tasks shuffled together (the upper bound, not continual)
It prints each method's accuracy matrix, average accuracy (ACC) and backward transfer (BWT),
for class-incremental testing (choose among all 10 digits) and task-incremental testing
(told which pair the image comes from). Saves results/task_11_continual.png.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import synthetic_digits, set_seed, results_dir

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
EPOCHS, BATCH, LR = 4, 64, 1e-3

Xtr_all, ytr_all = synthetic_digits(n_per_class=500, seed=0)
Xte_all, yte_all = synthetic_digits(n_per_class=200, seed=1)
Xtr_all, Xte_all = Xtr_all.reshape(len(Xtr_all), -1), Xte_all.reshape(len(Xte_all), -1)


def task_data(classes, train=True):
    X, y = (Xtr_all, ytr_all) if train else (Xte_all, yte_all)
    m = np.isin(y, classes)
    return torch.from_numpy(X[m]), torch.from_numpy(y[m])


TRAIN = [task_data(c, True) for c in TASKS]
TEST = [task_data(c, False) for c in TASKS]


def make_model():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


class Reservoir:
    """Keeps a uniform random sample of everything seen so far (Vitter 1985)."""

    def __init__(self, capacity, seed=0):
        self.capacity, self.seen = capacity, 0
        self.X, self.y = [], []
        self.rng = np.random.default_rng(seed)

    def add(self, X, y):
        for xi, yi in zip(X, y):
            self.seen += 1
            if len(self.X) < self.capacity:
                self.X.append(xi); self.y.append(yi)
            else:
                j = self.rng.integers(0, self.seen)   # keep the new item with probability capacity/seen
                if j < self.capacity:
                    self.X[j], self.y[j] = xi, yi

    def sample(self, n):
        idx = self.rng.integers(0, len(self.X), size=min(n, len(self.X)))
        return torch.stack([self.X[i] for i in idx]), torch.stack([self.y[i] for i in idx])


def fisher_diagonal(model, X, y, n=300):
    """Empirical Fisher: the mean squared gradient of log p(y | x) for each parameter."""
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    idx = torch.randperm(len(X))[:n]
    for i in idx:
        model.zero_grad()
        logp = F.log_softmax(model(X[i:i + 1]), dim=1)[0, y[i]]
        logp.backward()
        for f, p in zip(fisher, model.parameters()):
            f += p.grad.detach() ** 2
    return [f / len(idx) for f in fisher]


def evaluate(model):
    """Returns class-incremental and task-incremental accuracy on every task's test set."""
    model.eval()
    cil, til = [], []
    with torch.no_grad():
        for (X, y), classes in zip(TEST, TASKS):
            logits = model(X)
            cil.append((logits.argmax(1) == y).float().mean().item())
            masked = torch.full_like(logits, float("-inf"))
            masked[:, list(classes)] = logits[:, list(classes)]          # only this task's outputs
            til.append((masked.argmax(1) == y).float().mean().item())
    model.train()
    return cil, til


def run(method, lam=1e4, buffer=200, seed=0):
    set_seed(seed)
    model = make_model()
    memory = Reservoir(buffer, seed) if "replay" in method else None
    # EWC keeps one quadratic penalty per finished task k: lam/2 * sum_i F_k,i (theta_i - theta*_k,i)^2.
    # Summed over tasks this equals lam/2 * sum_i (A_i theta_i^2 - 2 B_i theta_i) plus a constant,
    # with A = sum_k F_k and B = sum_k F_k theta*_k, so two running vectors are all we need to store.
    A = B = None
    R_cil, R_til = np.zeros((5, 5)), np.zeros((5, 5))
    for t, (X, y) in enumerate(TRAIN):
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), BATCH):
                xb, yb = X[perm[i:i + BATCH]], y[perm[i:i + BATCH]]
                if memory is not None and memory.X:          # mix in old examples
                    xm, ym = memory.sample(BATCH)
                    xb_all, yb_all = torch.cat([xb, xm]), torch.cat([yb, ym])
                else:
                    xb_all, yb_all = xb, yb
                loss = F.cross_entropy(model(xb_all), yb_all)
                if "ewc" in method and A is not None:        # pull important weights back
                    theta = parameters_to_vector(model.parameters())
                    loss = loss + lam / 2 * (A * theta ** 2 - 2 * B * theta).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if memory is not None:
                    memory.add(xb, yb)
        if "ewc" in method:
            fisher = parameters_to_vector(fisher_diagonal(model, X, y))
            theta_star = parameters_to_vector(model.parameters()).detach()
            A = fisher if A is None else A + fisher
            B = fisher * theta_star if B is None else B + fisher * theta_star
        R_cil[t], R_til[t] = evaluate(model)
    return R_cil, R_til


def run_joint(seed=0):
    set_seed(seed)
    model = make_model()
    X = torch.cat([d[0] for d in TRAIN]); y = torch.cat([d[1] for d in TRAIN])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    cil, til = evaluate(model)
    return np.tile(cil, (5, 1)), np.tile(til, (5, 1))


def acc_bwt(R):
    T = len(R)
    acc = R[-1].mean()                                        # average accuracy after the last task
    bwt = np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])  # how much old tasks changed
    return acc, bwt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", action="store_true", help="sweep the EWC strength first")
    args = ap.parse_args()

    if args.lambdas:
        print("EWC strength sweep (task-incremental ACC / class-incremental ACC):")
        for lam in (1e2, 1e3, 1e4, 1e5, 1e6):
            Rc, Rt = run("ewc", lam=lam)
            print(f"  lambda {lam:>9,.0f}: TIL {acc_bwt(Rt)[0]:.3f}   CIL {acc_bwt(Rc)[0]:.3f}")

    t0 = time.time()
    results = {}
    # lambda for EWC alone came from the --lambdas sweep; with replay a much smaller value works best
    for name, lam in (("fine-tune", 0), ("ewc", 1e4), ("replay", 0), ("replay+ewc", 1e2)):
        results[name] = run(name, lam=lam)
    results["joint"] = run_joint()

    np.set_printoptions(precision=2, suppress=True)
    for name in ("fine-tune", "replay"):
        print(f"\nAccuracy matrix, {name}, class-incremental (row = after task i, column = task j):")
        print(results[name][0])
    print(f"\n{'method':<13}{'CIL ACC':>9}{'CIL BWT':>9}{'TIL ACC':>9}{'TIL BWT':>9}")
    for name, (Rc, Rt) in results.items():
        a, b = acc_bwt(Rc)
        at, bt = acc_bwt(Rt)
        print(f"{name:<13}{a:>9.3f}{b:>9.3f}{at:>9.3f}{bt:>9.3f}")
    print(f"({time.time() - t0:.0f} s)")

    colours = {"fine-tune": "#7F7F7F", "ewc": "#8172B2", "replay": "#3B6EA8", "replay+ewc": "#4C9A5F", "joint": "#E07B39"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, k, title in ((axes[0], 0, "Class-incremental (pick from 10 digits)"),
                         (axes[1], 1, "Task-incremental (told the pair)")):
        for name, R in results.items():
            seen_avg = [R[k][t, :t + 1].mean() for t in range(5)]
            ax.plot(range(1, 6), seen_avg, marker="o", color=colours[name], label=name,
                    ls="--" if name == "joint" else "-")
        ax.set_title(title, fontsize=10); ax.set_xlabel("tasks learned so far")
        ax.set_xticks(range(1, 6)); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean accuracy on tasks seen"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir() / "task_11_continual.png", dpi=150, bbox_inches="tight")
    print("saved", results_dir() / "task_11_continual.png")
