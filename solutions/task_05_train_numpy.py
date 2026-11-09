"""Task 5 solution: train the NumPy MLP on three-arm spirals and plot what it learned.

    python task_05_train_numpy.py

Mini-batch SGD with momentum, a 2-64-64-3 ReLU network, softmax cross-entropy.
Saves results/task_05_numpy_spirals.png (decision regions and loss curves).
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import make_spirals, train_val_test_split, plot_decision_boundary, results_dir


class NumpyMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.params = [[rng.normal(0, np.sqrt(2.0 / a), (b, a)), np.zeros(b)]
                       for a, b in zip(sizes[:-1], sizes[1:])]
        self.velocity = [[np.zeros_like(W), np.zeros_like(b)] for W, b in self.params]

    def forward(self, X):
        self.cache = [(None, X)]
        A = X
        for k, (W, b) in enumerate(self.params):
            Z = A @ W.T + b
            A = np.maximum(0, Z) if k < len(self.params) - 1 else Z
            self.cache.append((Z, A))
        return A

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        Zs = logits - logits.max(1, keepdims=True)
        P = np.exp(Zs) / np.exp(Zs).sum(1, keepdims=True)
        N = len(y)
        loss = -np.mean(np.log(P[np.arange(N), y] + 1e-12))
        dZ = P.copy()
        dZ[np.arange(N), y] -= 1.0
        dZ /= N
        grads = [None] * len(self.params)
        for k in reversed(range(len(self.params))):
            W, _ = self.params[k]
            grads[k] = [dZ.T @ self.cache[k][1], dZ.sum(0)]
            if k > 0:
                dZ = (dZ @ W) * (self.cache[k][0] > 0)
        return loss, grads

    def step(self, grads, lr=0.05, momentum=0.9):
        for (W, b), (gW, gb), v in zip(self.params, grads, self.velocity):
            v[0] = momentum * v[0] - lr * gW     # heavy-ball momentum
            v[1] = momentum * v[1] - lr * gb
            W += v[0]
            b += v[1]

    def predict(self, X):
        return self.forward(X).argmax(1)


def main():
    X, y = make_spirals(n=1500, arms=3, noise=0.2, seed=0)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(X, y, seed=0)
    X64 = lambda a: a.astype(np.float64)

    model = NumpyMLP([2, 64, 64, 3], seed=0)
    rng = np.random.default_rng(0)
    epochs, batch = 300, 64
    hist = {"train": [], "val": []}
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            loss, grads = model.loss_and_grads(X64(Xtr[idx]), ytr[idx])
            model.step(grads, lr=0.05 if epoch < 200 else 0.01)   # drop the step size near the end
        tr_loss, _ = model.loss_and_grads(X64(Xtr), ytr)
        va_loss, _ = model.loss_and_grads(X64(Xva), yva)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        if epoch in (0, 9, 49, 99, 199, epochs - 1):
            acc_tr = (model.predict(X64(Xtr)) == ytr).mean()
            acc_va = (model.predict(X64(Xva)) == yva).mean()
            print(f"epoch {epoch + 1:>3}: train loss {tr_loss:.4f}  train acc {acc_tr:.3f}  val acc {acc_va:.3f}")
    print(f"test accuracy: {(model.predict(X64(Xte)) == yte).mean():.3f}   ({time.time() - t0:.1f} s)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_decision_boundary(lambda G: model.predict(G.astype(np.float64)), Xtr, ytr, axes[0],
                           title="Decision regions after training")
    axes[1].plot(hist["train"], color="#3B6EA8", label="train")
    axes[1].plot(hist["val"], color="#E07B39", label="validation")
    axes[1].set_yscale("log"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cross-entropy")
    axes[1].grid(alpha=0.3); axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "task_05_numpy_spirals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
