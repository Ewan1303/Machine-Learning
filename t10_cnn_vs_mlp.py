"""Test 10: CNN vs MLP on shifted digits.

Trains three small networks on the same 16x16 digit images:
  - an MLP (multilayer perceptron) that sees the image as a flat list of 256 numbers,
  - a CNN whose convolution layers end in a flatten + linear head,
  - a CNN whose convolution layers end in global average pooling + linear head.
Training images are only ever moved by up to 1 pixel. The test then moves each test digit
0, 1, 2, 3 or 4 pixels (left, right, up or down, chosen at random) and measures accuracy.

Run:
    python t10_cnn_vs_mlp.py           # about a minute on a laptop CPU
    python t10_cnn_vs_mlp.py --quick   # 4 epochs instead of 10

What to look for: all three do well on digits that sit where the training digits sat.
As the shift grows, the MLP falls apart first, even though it has the most parameters.
The CNN with global average pooling holds on longest: its filters are shared across
positions (equivariance) and the pooling throws away where a feature was (invariance).
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

from mlguide_data import count_params, results_dir, set_seed, synthetic_digits


# --------------------------------------------------------------------------- models
class MLP(nn.Module):
    """Flatten the 16x16 image into 256 numbers, then two hidden layers of 128."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def conv_stack():
    """Three 3x3 convolution layers; the two max-pools halve the image twice (16 -> 8 -> 4)."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # (B,16,8,8)
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # (B,32,4,4)
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),                   # (B,64,4,4)
    )


class CNNFlatten(nn.Module):
    """Convolutions, then flatten all 64 x 4 x 4 numbers into a linear layer.
    The linear layer has a separate weight for every position, so it still cares *where*."""

    def __init__(self):
        super().__init__()
        self.features = conv_stack()
        self.head = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CNNGap(nn.Module):
    """Convolutions, then global average pooling: one number per channel, position forgotten."""

    def __init__(self):
        super().__init__()
        self.features = conv_stack()
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        h = self.features(x)            # (B, 64, 4, 4)
        h = h.mean(dim=(2, 3))          # global average pooling -> (B, 64)
        return self.head(h)


# --------------------------------------------------------------------------- shifting
def shift_images(X, pixels, seed=0):
    """Move every image `pixels` pixels left, right, up or down (random per image).
    Pixels that move off the edge are lost; the space left behind is filled with zeros."""
    if pixels == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    direction = rng.integers(0, 4, len(X))
    out = np.zeros_like(X)
    p = pixels
    # each slice pair copies the part of the image that stays on the canvas
    moves = {0: (np.s_[:, :, :, p:], np.s_[:, :, :, :-p]),   # right
             1: (np.s_[:, :, :, :-p], np.s_[:, :, :, p:]),   # left
             2: (np.s_[:, :, p:, :], np.s_[:, :, :-p, :]),   # down
             3: (np.s_[:, :, :-p, :], np.s_[:, :, p:, :])}   # up
    for d, (dst, src) in moves.items():
        idx = np.nonzero(direction == d)[0]
        block = np.zeros_like(X[idx])
        block[dst] = X[idx][src]
        out[idx] = block
    return out


# --------------------------------------------------------------------------- training
def train(model, X, y, epochs, batch=128, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    n = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    pred = model(torch.from_numpy(X)).argmax(1).numpy()
    return float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="4 epochs instead of 10")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    epochs = args.epochs or (4 if args.quick else 10)

    set_seed(0)
    t0 = time.time()
    # Training digits: random shifts of up to 1 pixel. Test digits: centred, then shifted below.
    X_train, y_train = synthetic_digits(n_per_class=600, seed=0, max_shift=1.0)
    X_test, y_test = synthetic_digits(n_per_class=200, seed=7, max_shift=0.0)

    models = {"MLP": MLP(), "CNN + flatten": CNNFlatten(), "CNN + global avg pool": CNNGap()}
    shifts = [0, 1, 2, 3, 4]
    results = {}
    for name, model in models.items():
        set_seed(0)
        start = time.time()
        train(model, X_train, y_train, epochs)
        accs = [accuracy(model, shift_images(X_test, s, seed=s), y_test) for s in shifts]
        results[name] = accs
        print(f"trained {name:<22s} in {time.time() - start:5.1f} s")

    print()
    print(f"{'model':<24s}{'params':>8s}" + "".join(f"{'shift ' + str(s):>9s}" for s in shifts))
    for name, model in models.items():
        row = "".join(f"{100 * a:8.1f}%" for a in results[name])
        print(f"{name:<24s}{count_params(model):>8,d}{row}")
    print(f"\ntotal run time {time.time() - t0:.0f} s")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(6.4, 4.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 2.6], hspace=0.35)
    k = int(np.nonzero(y_test == 4)[0][0])          # one example digit to show
    for j, s in enumerate(shifts):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(shift_images(X_test[k:k + 1], s, seed=1)[0, 0], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"shift {s} px", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    colours = {"MLP": "#C44E52", "CNN + flatten": "#E07B39", "CNN + global avg pool": "#3B6EA8"}
    for name, accs in results.items():
        label = f"{name} ({count_params(models[name]):,} params)"
        ax.plot(shifts, [100 * a for a in accs], "o-", color=colours[name], lw=1.8, label=label)
    ax.axhline(10, color="#7F7F7F", ls=":", lw=1)
    ax.text(0, 12, "chance", color="#7F7F7F", fontsize=8, ha="left")
    ax.set_xlabel("shift of the test digits (pixels); training digits moved at most 1 pixel", fontsize=9)
    ax.set_ylabel("test accuracy (%)", fontsize=9)
    ax.set_xticks(shifts)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.33), frameon=False)
    ax.tick_params(labelsize=8.5)
    out = results_dir() / "t10_cnn_vs_mlp.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
