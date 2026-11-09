"""Task 13 solution: the main permuted-pixels result of Kirkpatrick et al. (2017), in miniature.

    python task_13_ewc_paper.py
    python task_13_ewc_paper.py --sweep     # what different EWC strengths do

Three tasks share one 10-way output: A is the digits as they are, B and C are the same
digits with the pixels shuffled by two fixed permutations. One MLP learns A, then B, then C
with plain SGD. Three variants:
  SGD   no protection
  L2    a penalty pulling every weight back towards its old value, all weights equally
  EWC   the same pull, weighted by each weight's Fisher information
The paper's finding: SGD forgets A, L2 keeps A but struggles to learn B and C, EWC does both.
Saves results/task_13_ewc_paper.png, laid out like the paper's Figure 2A.
"""
import sys
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
from mlguide_data import synthetic_digits, permute_pixels, set_seed, results_dir

EPOCHS, BATCH, LR = 5, 64, 0.1
Xtr, ytr = synthetic_digits(n_per_class=300, seed=0)
Xte, yte = synthetic_digits(n_per_class=100, seed=1)
flat = lambda X: torch.from_numpy(X.reshape(len(X), -1))
TASKS = []
for name, perm_seed in (("A", None), ("B", 11), ("C", 22)):
    tr = Xtr if perm_seed is None else permute_pixels(Xtr, perm_seed)
    te = Xte if perm_seed is None else permute_pixels(Xte, perm_seed)   # same permutation for both
    TASKS.append((name, flat(tr), torch.from_numpy(ytr), flat(te), torch.from_numpy(yte)))


def make_model():
    return nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def fisher(model, X, y, n=500):
    f = torch.zeros_like(parameters_to_vector(model.parameters()))
    for i in torch.randperm(len(X))[:n]:
        model.zero_grad()
        F.log_softmax(model(X[i:i + 1]), 1)[0, y[i]].backward()
        f += parameters_to_vector([p.grad for p in model.parameters()]) ** 2
    return f / n


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def run(method, lam):
    set_seed(0)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    A = B = None                      # running sums of importance and importance * old weights
    curves = {name: [] for name, *_ in TASKS}
    for t, (name, X, y, _, _) in enumerate(TASKS):
        for _ in range(EPOCHS):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), BATCH):
                idx = perm[i:i + BATCH]
                loss = F.cross_entropy(model(X[idx]), y[idx])
                if A is not None and method != "SGD":
                    theta = parameters_to_vector(model.parameters())
                    loss = loss + lam / 2 * (A * theta ** 2 - 2 * B * theta).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            for tn, *_rest in TASKS:                            # test every task after every epoch
                k = [n for n, *_ in TASKS].index(tn)
                curves[tn].append(accuracy(model, TASKS[k][3], TASKS[k][4]) if k <= t else np.nan)
        if method != "SGD":
            theta_star = parameters_to_vector(model.parameters()).detach()
            imp = fisher(model, X, y) if method == "EWC" else torch.ones_like(theta_star)
            A = imp if A is None else A + imp
            B = imp * theta_star if B is None else B + imp * theta_star
    return curves


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        # With plain SGD the penalty adds curvature lam * F_i to each weight's loss; once
        # LR * lam * F_i passes 2 the updates overshoot and training diverges (see 1.3).
        for lam in (30, 100, 200, 1000):
            cur = run("EWC", lam)
            print(f"EWC lambda {lam:>5}: A {cur['A'][-1]:.3f}  B {cur['B'][-1]:.3f}  C {cur['C'][-1]:.3f}")
        sys.exit(0)
    settings = {"SGD": 0.0, "L2": 0.1, "EWC": 100.0}
    results = {m: run(m, lam) for m, lam in settings.items()}
    print(f"{'method':<6}{'lambda':>8}{'A at end':>10}{'B at end':>10}{'C at end':>10}{'mean':>8}")
    for m, cur in results.items():
        final = [cur[n][-1] for n in "ABC"]
        print(f"{m:<6}{settings[m]:>8g}{final[0]:>10.3f}{final[1]:>10.3f}{final[2]:>10.3f}{np.mean(final):>8.3f}")
    print("(A was learned first. Each task gets", EPOCHS, "epochs of SGD.)")

    colours = {"SGD": "#3B6EA8", "L2": "#4C9A5F", "EWC": "#C44E52"}
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.2), sharex=True)
    x = np.arange(1, 3 * EPOCHS + 1)
    for ax, n in zip(axes, "ABC"):
        for m, cur in results.items():
            ax.plot(x, cur[n], color=colours[m], lw=1.8, label=m)
        for b in (EPOCHS + 0.5, 2 * EPOCHS + 0.5):
            ax.axvline(b, color="#7F7F7F", ls=":", lw=1)
        ax.set_ylabel(f"task {n}", fontsize=9); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, fontsize=8.5, loc="lower left")
    axes[0].set_title("Test accuracy while training on A, then B, then C", fontsize=10)
    axes[-1].set_xlabel("training epoch")
    fig.tight_layout()
    fig.savefig(results_dir() / "task_13_ewc_paper.png", dpi=150, bbox_inches="tight")
    print("saved", results_dir() / "task_13_ewc_paper.png")
