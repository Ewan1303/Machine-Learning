"""Test 22: A Hopfield network remembers.

A Hopfield network stores memories in its weights and recalls a whole memory from a
damaged or partial cue: the network's state rolls downhill on an energy surface into
the nearest stored pattern. This script:

  1. stores a few digit images (16 x 16 = 256 binary neurons) and recalls them from
     cues with 20% of the pixels flipped, using
       the classical network   W = (1/N) sum of x x^T  (Hebb's rule), one neuron at a time
       the modern network      x_new = X^T softmax(beta X x)  (Ramsauer et al. 2021),
                               which is the same computation as attention
  2. measures capacity: how well each network recalls random patterns as you store more
     of them, compared with the classical limit of about 0.138 N (Amit et al. 1985).

Run:   python t22_hopfield.py          (about 10 s)

Look for: the classical network blends similar memories together and breaks down near
0.14 N patterns; the modern one keeps recalling cleanly far beyond N patterns.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlguide_data import results_dir, synthetic_digits

DIGITS = (0, 1, 4, 7)   # the digit images stored in part 1
CUE_FLIP = 0.20         # fraction of pixels flipped in the part-1 cues
BETA = 8.0              # sharpness of the modern network's softmax, in units of 1/sqrt(N)


def hebbian_weights(patterns):
    """Classical storage rule. patterns has shape (P, N) with entries +1 or -1."""
    N = patterns.shape[1]
    W = patterns.T @ patterns / N
    np.fill_diagonal(W, 0.0)          # no neuron feeds back onto itself
    return W


def recall_classical(W, cues, rng, max_sweeps=20):
    """Asynchronous updates: one neuron at a time, in random order, until nothing changes.

    cues has shape (T, N); all T recalls run side by side.
    """
    s = cues.copy()
    N = W.shape[0]
    for _ in range(max_sweeps):
        changed = False
        for i in rng.permutation(N):
            h = s @ W[:, i]                               # input to neuron i
            new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, s[:, i]))
            changed |= bool(np.any(new != s[:, i]))
            s[:, i] = new
        if not changed:
            break
    return s


def recall_modern(patterns, cues, beta, steps=1):
    """Modern Hopfield update: a softmax-weighted average of the stored patterns."""
    s = cues.copy()
    for _ in range(steps):
        scores = beta * s @ patterns.T                     # similarity to every memory
        scores -= scores.max(1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(1, keepdims=True)                       # attention weights
        s = np.sign(p @ patterns)
        s[s == 0] = 1.0
    return s


def corrupt(patterns, frac, rng):
    flip = rng.random(patterns.shape) < frac
    return np.where(flip, -patterns, patterns)


def energy(W, s):
    return -0.5 * np.einsum("ti,ij,tj->t", s, W, s)


def main():
    t0 = time.time()
    rng = np.random.default_rng(22)

    # ---- 1. digit memories --------------------------------------------------------
    X, y = synthetic_digits(n_per_class=1, seed=22, max_shift=0, max_rotate=0,
                            scale_range=(1.0, 1.0), noise=0.0, cache=False)
    order = [list(y).index(d) for d in DIGITS]
    mem = np.where(X[order].reshape(len(order), -1) > 0.2, 1.0, -1.0)   # (digits, 256)
    N = mem.shape[1]
    W = hebbian_weights(mem)
    cues = corrupt(mem, CUE_FLIP, rng)
    out_c = recall_classical(W, cues, rng)
    out_m = recall_modern(mem, cues, beta=BETA / np.sqrt(N))
    ov = lambda a, b: (a * b).sum(1) / a.shape[1]         # overlap: 1 = identical
    corr = np.corrcoef(mem)[np.triu_indices(len(mem), 1)]
    print(f"Digit memories: {len(mem)} images of {N} pixels; "
          f"mean correlation between stored images {corr.mean():.2f}")
    print(f"  {'digit':6s} {'cue overlap':>12s} {'classical':>10s} {'modern':>8s}")
    for k, d in enumerate(DIGITS):
        print(f"  {d:<6d} {ov(cues, mem)[k]:12.2f} {ov(out_c, mem)[k]:10.2f} {ov(out_m, mem)[k]:8.2f}")
    print(f"  energy of the cue vs the classical result: "
          f"{energy(W, cues).mean():.1f} -> {energy(W, out_c).mean():.1f} (recall only goes downhill)")

    # ---- 2. capacity with random patterns -----------------------------------------
    Nc = 300
    loads = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.22, 0.30, 0.5, 1.0, 2.0])
    res_c, res_m = [], []
    for a in loads:
        P = max(1, int(round(a * Nc)))
        pats = rng.choice([-1.0, 1.0], size=(P, Nc))
        test = pats[: min(P, 40)]
        cue = corrupt(test, 0.10, rng)
        if a <= 0.5:
            out = recall_classical(hebbian_weights(pats), cue, rng)
            res_c.append(ov(out, test).mean())
        else:
            res_c.append(np.nan)                          # far past breaking: skip the slow part
        res_m.append(ov(recall_modern(pats, cue, beta=BETA / np.sqrt(Nc)), test).mean())
    print(f"\nCapacity, N = {Nc} neurons, cues with 10% of bits flipped "
          f"(final overlap with the right memory, 1.00 = perfect):")
    print(f"  {'patterns / N':>13s} {'patterns':>9s} {'classical':>10s} {'modern':>8s}")
    for a, c, m in zip(loads, res_c, res_m):
        cs = "     -" if np.isnan(c) else f"{c:10.3f}"
        print(f"  {a:13.2f} {int(round(a * Nc)):9d} {cs:>10s} {m:8.3f}")

    # ---- figure -------------------------------------------------------------------
    rows = len(DIGITS)
    fig = plt.figure(figsize=(9.4, max(3.6, 0.9 * rows)))
    gs = fig.add_gridspec(rows, 6, width_ratios=[1, 1, 1, 1, 0.75, 4.2], wspace=0.08, hspace=0.12)
    titles = ["stored", f"cue ({100 * CUE_FLIP:.0f}%\nflipped)", "classical\nrecall", "modern\nrecall"]
    for r in range(rows):
        for c, img in enumerate([mem[r], cues[r], out_c[r], out_m[r]]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img.reshape(16, 16), cmap="gray_r", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    ax = fig.add_subplot(gs[:, 5])
    ax.plot(loads[: len(res_c)], res_c, "o-", color="#C44E52", lw=1.8, ms=4, label="classical (Hebbian)")
    ax.plot(loads, res_m, "s-", color="#3B6EA8", lw=1.8, ms=4, label="modern (softmax)")
    ax.axvline(0.138, color="k", ls=":", lw=1)
    ax.text(0.147, 0.25, "0.138 N", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.set_xlabel("patterns stored / neurons (log scale)")
    ax.set_ylabel("overlap after recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Recall from 10% corrupted cues, N = {Nc}", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    out = results_dir() / "t22_hopfield.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out.name}. Run time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
