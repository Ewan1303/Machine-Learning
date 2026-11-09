"""Test 11: How long can a network remember?

Each training example is a sequence of T random bits (+1 or -1), one per time step, plus a
marker that is 1 only at the first step. After reading the whole sequence the network must
say what the FIRST bit was. Every later bit is a distractor, so the network has to store one
bit for T steps and protect it from everything that arrives afterwards. Chance is 50%.

Five recurrent networks with the same hidden size (32) are trained on fresh random sequences
for each length T:
  - a plain (Elman) RNN,
  - a GRU and an LSTM with their usual starting settings (PyTorch defaults; for the LSTM the
    common advice of a forget-gate bias of 1),
  - the same GRU and LSTM with the gate that controls forgetting started at "keep"
    (gate bias 3, so the gate starts at about 0.95 instead of 0.5 or 0.73).

The script also measures, before any training, how much gradient from the loss reaches the
first input (the size of dLoss/dx_0). That is the signal the network needs in order to learn
to store the bit. For the plain RNN it shrinks exponentially with T: the vanishing gradient.

Run:
    python t11_memory_length.py           # about 1-2 minutes on a laptop CPU
    python t11_memory_length.py --long    # also T = 160 (a few minutes more)

What to look for: every network solves short sequences. The plain RNN falls to chance by
T = 20. The gated networks with the forget gate started at "keep" hold the bit for much longer,
and their gradient at step 0 is many orders of magnitude larger. With the default gate settings
the gated networks get only a little further than the plain RNN within this training budget.
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

from mlguide_data import count_params, results_dir, set_seed

torch.set_num_threads(1)   # these layers are tiny; one thread is faster than several


def make_batch(T, batch, rng):
    """x has shape (batch, T, 2): channel 0 is a random +-1 bit, channel 1 marks step 0.
    y is the class of the first bit (1 if it was +1, else 0)."""
    bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(batch, T))
    marker = np.zeros((batch, T), dtype=np.float32)
    marker[:, 0] = 1.0
    x = np.stack([bits, marker], axis=-1)
    y = (bits[:, 0] > 0).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class Recaller(nn.Module):
    """A recurrent layer reads the sequence; a linear layer reads out the final state.

    keep_bias sets the starting bias of the gate that decides how much of the old state is
    kept: the LSTM's forget gate, or the GRU's update gate z (in PyTorch's GRU,
    h_new = (1 - z) * candidate + z * h_old, so a large z keeps the old state).
    PyTorch stores two bias vectors per layer that are added together, so each gets half.
    Gate order in PyTorch: LSTM (input, forget, cell, output); GRU (reset, update, new).
    Both keep-gates are therefore the second block of `hidden` entries.
    """

    def __init__(self, kind, keep_bias=None, hidden=32):
        super().__init__()
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = cls(input_size=2, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)
        if keep_bias is not None:
            for name, p in self.rnn.named_parameters():
                if name.startswith("bias"):
                    with torch.no_grad():
                        p[hidden:2 * hidden].fill_(keep_bias / 2)

    def forward(self, x):
        h, _ = self.rnn(x)            # h: (batch, T, hidden), the state after every step
        return self.out(h[:, -1])     # classify from the state after the last step


MODELS = [  # label, kind, keep-gate bias, colour, line style
    ("RNN", "RNN", None, "#C44E52", "-"),
    ("GRU, default start", "GRU", None, "#4C9A5F", "-"),
    ("LSTM, forget bias 1", "LSTM", 1.0, "#3B6EA8", "-"),
    ("GRU, update bias 3", "GRU", 3.0, "#4C9A5F", "--"),
    ("LSTM, forget bias 3", "LSTM", 3.0, "#3B6EA8", "--"),
]


def grad_reaching_first_input(kind, keep_bias, T, seed=0, batch=256):
    """Average size of dLoss/dx_0 at initialisation (before any training)."""
    torch.manual_seed(seed)
    model = Recaller(kind, keep_bias)
    x, y = make_batch(T, batch, np.random.default_rng(seed))
    x.requires_grad_(True)
    F.cross_entropy(model(x), y).backward()
    return x.grad[:, 0].norm(dim=1).mean().item()


def train_and_test(kind, keep_bias, T, steps, batch=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000 * T)
    model = Recaller(kind, keep_bias)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x, y = make_batch(T, batch, rng)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against exploding gradients
        opt.step()
    with torch.no_grad():
        x, y = make_batch(T, 2000, np.random.default_rng(12345))
        acc = (model(x).argmax(1) == y).float().mean().item()
    return acc, count_params(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="also test T = 160")
    ap.add_argument("--steps", type=int, default=300, help="training steps per run")
    args = ap.parse_args()
    lengths = [5, 10, 20, 40, 80] + ([160] if args.long else [])
    set_seed(0)
    t0 = time.time()

    acc, grad, params = {}, {}, {}
    for label, kind, kb, _, _ in MODELS:
        acc[label], grad[label] = [], []
        for T in lengths:
            grad[label].append(grad_reaching_first_input(kind, kb, T))
            a, params[label] = train_and_test(kind, kb, T, args.steps)
            acc[label].append(a)
        print(f"{label:<22s} done after {time.time() - t0:5.0f} s")

    labels = [m[0] for m in MODELS]
    print("\nTest accuracy recalling the first bit (chance = 50%)")
    print(f"{'T':>5s}" + "".join(f"{lab:>22s}" for lab in labels))
    for i, T in enumerate(lengths):
        print(f"{T:>5d}" + "".join(f"{100 * acc[lab][i]:21.1f}%" for lab in labels))
    print("\nSize of dLoss/dx_0 at initialisation")
    print(f"{'T':>5s}" + "".join(f"{lab:>22s}" for lab in labels))
    for i, T in enumerate(lengths):
        print(f"{T:>5d}" + "".join(f"{grad[lab][i]:22.1e}" for lab in labels))
    print("\nparameters: " + ", ".join(f"{lab} {params[lab]:,}" for lab in labels))
    print(f"total run time {time.time() - t0:.0f} s")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for label, kind, kb, colour, ls in MODELS:
        axes[0].plot(lengths, [100 * a for a in acc[label]], marker="o", ms=4, ls=ls, color=colour,
                     lw=1.6, label=label)
        axes[1].plot(lengths, grad[label], marker="o", ms=4, ls=ls, color=colour, lw=1.6, label=label)
    axes[0].axhline(50, color="#7F7F7F", ls=":", lw=1)
    axes[0].text(lengths[-1], 52, "chance", color="#7F7F7F", fontsize=8, ha="right")
    axes[0].set_ylim(40, 103)
    axes[0].set_ylabel("test accuracy (%)", fontsize=9)
    axes[0].set_title("Recall the first bit after T steps", fontsize=9.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("size of dLoss/dx$_0$ (log scale)", fontsize=9)
    axes[1].set_title("Gradient reaching step 0 before training", fontsize=9.5)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([str(t) for t in lengths])
        ax.set_xlabel("sequence length T", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    out = results_dir() / "t11_memory_length.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
