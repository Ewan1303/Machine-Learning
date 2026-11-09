"""Test 12: Attention as lookup.

Each example is a small dictionary: 8 (key, value) pairs, where keys are 8 different symbols
out of 16 and values are symbols out of 16, followed by a query that is one of the 8 keys.
The target is the value stored under that key. This is exactly what attention is built for:
compare the query with every key, put almost all the weight on the match, read out its value.

Two models are trained on fresh random dictionaries:
  - a single attention layer (about 4.6 thousand parameters), written from scratch below and
    checked against PyTorch's own scaled dot-product attention;
  - an MLP that sees the same information as one long flat vector (about 140 thousand parameters).

Run:
    python t12_attention.py           # about 30-60 s on a laptop CPU
    python t12_attention.py --quick   # fewer training steps

What to look for: the attention model reaches close to 100% while the much bigger MLP lags far
behind, and the plotted attention weights sit on the matching key.
"""
import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import count_params, results_dir, set_seed

N_KEYS, N_VALUES, N_PAIRS = 16, 16, 8


def make_batch(batch, rng):
    """keys (B, 8) distinct per row, values (B, 8), query position (B,), query key (B,), target (B,)."""
    keys = np.argsort(rng.random((batch, N_KEYS)), axis=1)[:, :N_PAIRS]   # 8 distinct keys per row
    values = rng.integers(0, N_VALUES, (batch, N_PAIRS))
    pos = rng.integers(0, N_PAIRS, batch)
    query = keys[np.arange(batch), pos]
    target = values[np.arange(batch), pos]
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.int64))
    return t(keys), t(values), t(pos), t(query), t(target)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention written out step by step.
    Q: (B, n, d_k), K: (B, m, d_k), V: (B, m, d_v) -> output (B, n, d_v), weights (B, n, m)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)    # similarity of every query with every key
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # blocked positions get zero weight
    weights = torch.softmax(scores, dim=-1)              # each row sums to 1
    return weights @ V, weights                           # weighted average of the values


class AttentionLookup(nn.Module):
    """One token per stored pair (key embedding + value embedding); the query attends over them."""

    def __init__(self, d=32):
        super().__init__()
        self.key_emb = nn.Embedding(N_KEYS, d)     # shared by stored keys and the query
        self.val_emb = nn.Embedding(N_VALUES, d)
        self.W_q = nn.Linear(d, d, bias=False)     # what am I looking for?
        self.W_k = nn.Linear(d, d, bias=False)     # what does each slot contain?
        self.W_v = nn.Linear(d, d, bias=False)     # what does each slot hand back?
        self.out = nn.Linear(d, N_VALUES)

    def forward(self, keys, values, query):
        tokens = self.key_emb(keys) + self.val_emb(values)   # (B, 8, d): key and value superimposed
        Q = self.W_q(self.key_emb(query)).unsqueeze(1)       # (B, 1, d)
        K, V = self.W_k(tokens), self.W_v(tokens)            # (B, 8, d) each
        ctx, w = attention(Q, K, V)                          # (B, 1, d), (B, 1, 8)
        return self.out(ctx.squeeze(1)), w.squeeze(1)


class FlatMLP(nn.Module):
    """The same information as one flat one-hot vector: 8 x (16 + 16) + 16 = 272 inputs."""

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_PAIRS * (N_KEYS + N_VALUES) + N_KEYS, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N_VALUES))

    def forward(self, keys, values, query):
        B = keys.shape[0]
        x = torch.cat([F.one_hot(keys, N_KEYS).float().reshape(B, -1),
                       F.one_hot(values, N_VALUES).float().reshape(B, -1),
                       F.one_hot(query, N_KEYS).float()], dim=1)
        return self.net(x), None


def check_against_pytorch():
    """Our attention() should match torch's F.scaled_dot_product_attention to rounding error."""
    torch.manual_seed(0)
    Q, K, V = torch.randn(4, 5, 16), torch.randn(4, 7, 16), torch.randn(4, 7, 16)
    ours, _ = attention(Q, K, V)
    theirs = F.scaled_dot_product_attention(Q, K, V)
    return (ours - theirs).abs().max().item()


def run(model, steps, rng, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    test = make_batch(2000, np.random.default_rng(999))
    # evaluate often early on (where the attention model learns) and every 100 steps after
    eval_at = set(range(0, 201, 10)) | set(range(300, steps + 1, 100))
    curve = []
    for step in range(0, steps + 1):
        if step > 0:
            keys, values, pos, query, target = make_batch(128, rng)
            logits, _ = model(keys, values, query)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step in eval_at:
            with torch.no_grad():
                logits, _ = model(test[0], test[1], test[3])
                curve.append((step, (logits.argmax(1) == test[4]).float().mean().item()))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    steps = args.steps or (800 if args.quick else 2000)
    set_seed(0)
    t0 = time.time()

    diff = check_against_pytorch()
    print(f"largest difference from F.scaled_dot_product_attention: {diff:.1e}")

    torch.manual_seed(0)
    att = AttentionLookup()
    curve_att = run(att, steps, np.random.default_rng(1))
    torch.manual_seed(0)
    mlp = FlatMLP()
    curve_mlp = run(mlp, steps, np.random.default_rng(1))

    print(f"\n{'step':>6s}{'attention':>12s}{'MLP':>9s}")
    for (s, a), (_, m) in zip(curve_att, curve_mlp):
        if s in (0, 20, 50, 100, 200, 500, 1000, 2000) or s == steps:
            print(f"{s:>6d}{100 * a:11.1f}%{100 * m:8.1f}%")
    print(f"\nparameters: attention {count_params(att):,}, MLP {count_params(mlp):,}")

    # where does the trained attention look?
    keys, values, pos, query, target = make_batch(8, np.random.default_rng(5))
    with torch.no_grad():
        logits, w = att(keys, values, query)
    on_match = w[torch.arange(8), pos].mean().item()
    print(f"average attention weight on the matching key (8 examples): {on_match:.3f}")
    print(f"total run time {time.time() - t0:.0f} s")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(*zip(*[(s, 100 * a) for s, a in curve_att]), color="#3B6EA8", lw=1.8,
            label=f"attention ({count_params(att):,} params)")
    ax.plot(*zip(*[(s, 100 * a) for s, a in curve_mlp]), color="#C44E52", lw=1.8,
            label=f"MLP ({count_params(mlp):,} params)")
    ax.axhline(100 / N_VALUES, color="#7F7F7F", ls=":", lw=1)
    ax.text(steps, 100 / N_VALUES + 2, "chance", color="#7F7F7F", fontsize=8, ha="right")
    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("test accuracy (%)", fontsize=9)
    ax.set_ylim(0, 102)
    ax.set_title("Look up the value stored under a key", fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.tick_params(labelsize=8.5)
    ax = axes[1]
    im = ax.imshow(w.numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(8):
        ax.add_patch(plt.Rectangle((pos[r].item() - 0.5, r - 0.5), 1, 1, fill=False, ec="#E07B39", lw=1.6))
    ax.set_xticks(range(N_PAIRS))
    ax.set_xticklabels([str(i + 1) for i in range(N_PAIRS)], fontsize=8)
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"query {r + 1}" for r in range(8)], fontsize=7.5)
    ax.set_xlabel("stored pair", fontsize=9)
    ax.set_title("Attention weights (box = correct key)", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    out = results_dir() / "t12_attention.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
