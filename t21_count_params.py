"""Test 21: counting parameters by hand and checking against PyTorch.

Every layer type has a simple formula for its number of learnable parameters.
This script works each one out by hand, asks PyTorch for the real count, and
prints both side by side so you can see they agree. It then compares layers that
all map 256 numbers to 256 numbers, to show how differently they spend parameters.

Run:
    python t21_count_params.py

Look for: every row says "yes" in the match column, and the bar chart showing a
transformer layer or a 3x3 convolution costs far more than a plain linear layer
of the same width.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch.nn as nn

from mlguide_data import count_params, results_dir, set_seed


def n_params(module):
    """Learnable parameters only (what the optimiser updates)."""
    return count_params(module, trainable_only=True)


def n_buffers(module):
    """Non-learnable saved state, such as BatchNorm running statistics."""
    return sum(b.numel() for b in module.buffers())


def main():
    set_seed(0)
    t0 = time.time()
    rows = []  # (layer, formula text, formula value, pytorch count)

    def check(name, formula, value, module):
        rows.append((name, formula, value, n_params(module)))

    # Dense (fully connected) layers: one weight per input-output pair, one bias per output
    check("Linear(784, 256)", "784*256 + 256", 784 * 256 + 256, nn.Linear(784, 256))
    check("Linear(784, 256, bias=False)", "784*256", 784 * 256, nn.Linear(784, 256, bias=False))

    # Convolutions: one k x k kernel per (input channel, output channel) pair, shared over positions
    check("Conv2d(3, 64, k=3)", "3*3*3*64 + 64", 3 * 3 * 3 * 64 + 64, nn.Conv2d(3, 64, 3))
    check("Conv2d(64, 128, k=3)", "3*3*64*128 + 128", 3 * 3 * 64 * 128 + 128, nn.Conv2d(64, 128, 3))
    check("Conv1d(8, 32, k=5)", "5*8*32 + 32", 5 * 8 * 32 + 32, nn.Conv1d(8, 32, 5))
    depthwise = nn.Sequential(nn.Conv2d(64, 64, 3, groups=64), nn.Conv2d(64, 128, 1))
    check("Depthwise separable 64->128, k=3", "(3*3*64 + 64) + (64*128 + 128)",
          (3 * 3 * 64 + 64) + (64 * 128 + 128), depthwise)

    # Lookup tables and normalisation
    check("Embedding(10000, 128)", "10000*128", 10000 * 128, nn.Embedding(10000, 128))
    check("LayerNorm(256)", "2*256", 2 * 256, nn.LayerNorm(256))
    bn = nn.BatchNorm2d(64)
    check("BatchNorm2d(64)", "2*64 (+ buffers)", 2 * 64, bn)

    # Recurrent layers. PyTorch keeps two bias vectors (b_ih and b_hh) per gate.
    x, h = 32, 128
    check("RNN(32, 128)", "h*x + h*h + 2h", h * x + h * h + 2 * h, nn.RNN(x, h))
    check("GRU(32, 128)", "3(h*x + h*h + 2h)", 3 * (h * x + h * h + 2 * h), nn.GRU(x, h))
    check("LSTM(32, 128)", "4(h*x + h*h + 2h)", 4 * (h * x + h * h + 2 * h), nn.LSTM(x, h))
    check("LSTM(32, 128, layers=2)", "4(h(x+h)+2h) + 4(h(h+h)+2h)",
          4 * (h * (x + h) + 2 * h) + 4 * (h * (h + h) + 2 * h), nn.LSTM(x, h, num_layers=2))

    # Attention: query, key, value and output projections, each d x d plus a bias
    d = 256
    check("MultiheadAttention(256, 8 heads)", "4d^2 + 4d", 4 * d * d + 4 * d,
          nn.MultiheadAttention(d, 8))
    check("TransformerEncoderLayer(256, ff=1024)", "12d^2 + 13d", 12 * d * d + 13 * d,
          nn.TransformerEncoderLayer(d, 8, dim_feedforward=4 * d))

    # A whole model: a 784-256-128-10 multilayer perceptron
    mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, 10))
    check("MLP 784-256-128-10", "sum over layers", (784 * 256 + 256) + (256 * 128 + 128) + (128 * 10 + 10), mlp)

    print(f"{'layer':40s} {'formula':32s} {'by hand':>10s} {'PyTorch':>10s}  match")
    print("-" * 102)
    for name, formula, value, real in rows:
        print(f"{name:40s} {formula:32s} {value:10,d} {real:10,d}  {'yes' if value == real else 'NO'}")
    print(f"\nBatchNorm2d(64) also stores {n_buffers(bn)} buffer values "
          "(running mean, running variance, batch counter) that are saved but not trained.")

    # Same input and output width (256), different layer types
    w = 256
    compare = [
        ("Linear", nn.Linear(w, w)),
        ("Depthwise sep. conv 3x3", nn.Sequential(nn.Conv2d(w, w, 3, groups=w), nn.Conv2d(w, w, 1))),
        ("RNN", nn.RNN(w, w)),
        ("Conv1d k=3", nn.Conv1d(w, w, 3)),
        ("Multi-head attention", nn.MultiheadAttention(w, 8)),
        ("GRU", nn.GRU(w, w)),
        ("LSTM", nn.LSTM(w, w)),
        ("Conv2d 3x3", nn.Conv2d(w, w, 3)),
        ("Transformer layer", nn.TransformerEncoderLayer(w, 8, dim_feedforward=4 * w)),
    ]
    names = [c[0] for c in compare]
    counts = [n_params(c[1]) for c in compare]
    print("\nLayers that all map 256 features to 256 features:")
    for nme, c in zip(names, counts):
        print(f"  {nme:26s} {c:9,d}  ({c / counts[0]:.1f}x a Linear layer)")

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    colours = ["#3B6EA8"] * len(names)
    bars = ax.barh(names, counts, color=colours)
    for b, c in zip(bars, counts):
        ax.text(c * 1.02, b.get_y() + b.get_height() / 2, f"{c:,}", va="center", fontsize=8)
    ax.set_xlabel("learnable parameters (width 256 in and out)", fontsize=9)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = results_dir() / "t21_count_params.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
