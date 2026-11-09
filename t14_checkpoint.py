"""Test 14: Saving and resuming exactly.

A network's long-term memory is its weights, but a training run remembers far more
than that: the optimiser's running averages, where the learning-rate schedule is,
which random numbers come next (dropout masks, data order) and BatchNorm's running
statistics. This script trains the same small network four ways:

  uninterrupted    8 epochs in one go
  full resume      4 epochs, save a complete checkpoint, throw everything away,
                   rebuild from a different seed, load, train 4 more epochs
  weights only     same, but only the model's weights are saved and loaded
  no RNG restore   weights, optimiser and scheduler restored, random state not

and then shows the BatchNorm/dropout trap: evaluating without model.eval().

Run:   python t14_checkpoint.py        (about 15 s)

Look for: the full resume matches the uninterrupted run exactly (difference 0.0);
weights only gives a loss jump right after the restart; forgetting model.eval()
lowers accuracy and silently changes BatchNorm's stored statistics.
"""
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import count_params, results_dir, set_seed, synthetic_digits, train_val_test_split

EPOCHS, HALF, BATCH, LR = 8, 4, 64, 1e-2


def make_model():
    """A small MLP for 16x16 digit images, with BatchNorm and dropout on purpose."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 10),
    )


def build(seed, steps_per_epoch):
    """Everything a run needs, created from one seed (as a fresh program would)."""
    set_seed(seed)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    data_gen = torch.Generator().manual_seed(seed)   # controls the shuffling order only
    return model, opt, sched, data_gen


def train_epochs(model, opt, sched, data_gen, X, y, n_epochs):
    """Plain mini-batch training. Returns the loss of every step."""
    losses = []
    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(len(X), generator=data_gen)
        for i in range(0, len(X) - BATCH + 1, BATCH):    # drop the last partial batch
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
    return losses


def numpy_state_plain():
    """NumPy's RNG state with the array turned into a list, so weights_only loading accepts it."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return (name, keys.tolist(), pos, has_gauss, cached)


def save_checkpoint(path, model, opt, sched, data_gen, epoch):
    state = {
        "model": model.state_dict(),          # weights and buffers (BatchNorm running stats)
        "optimizer": opt.state_dict(),        # Adam's moment estimates and step counts
        "scheduler": sched.state_dict(),      # position in the learning-rate schedule
        "epoch": epoch,
        "rng": {
            "python": random.getstate(),
            "numpy": numpy_state_plain(),
            "torch": torch.get_rng_state(),   # dropout masks are drawn from this
            "data_order": data_gen.get_state(),
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic rename: a crash never leaves a half-written file


def restore_rng(rng):
    random.setstate(rng["python"])
    name, keys, pos, has_gauss, cached = rng["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(rng["torch"])


@torch.no_grad()
def accuracy(model, X, y, batch=256):
    correct = 0
    for i in range(0, len(X), batch):
        correct += (model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(X)


def main():
    t0 = time.time()
    # Harder digits (bigger shifts, rotations and noise) so the loss stays visible.
    X, y = synthetic_digits(n_per_class=400, seed=14, max_shift=2.5, max_rotate=25, noise=0.2)
    (Xtr, ytr), _, (Xte, yte) = train_val_test_split(X, y, val=0.1, test=0.2, seed=14)
    Xtr, ytr, Xte, yte = map(torch.from_numpy, (Xtr, ytr, Xte, yte))
    steps = len(Xtr) // BATCH
    ckpt = results_dir() / "t14_checkpoint.pt"

    # 1. Uninterrupted run: 8 epochs in one go.
    model, opt, sched, gen = build(seed=0, steps_per_epoch=steps)
    loss_full = train_epochs(model, opt, sched, gen, Xtr, ytr, EPOCHS)
    model_uninterrupted = model

    # 2. First half of an identical run, then a complete checkpoint.
    model, opt, sched, gen = build(seed=0, steps_per_epoch=steps)
    loss_first = train_epochs(model, opt, sched, gen, Xtr, ytr, HALF)
    save_checkpoint(ckpt, model, opt, sched, gen, epoch=HALF)
    torch.save(model.state_dict(), results_dir() / "t14_weights_only.pt")
    del model, opt, sched, gen    # "the program stops"

    def resume(restore_optimiser, restore_rng_state):
        # A new program: different seed, freshly built objects.
        model, opt, sched, gen = build(seed=999, steps_per_epoch=steps)
        state = torch.load(ckpt, weights_only=True)   # safe loading: no arbitrary code
        model.load_state_dict(state["model"])
        if restore_optimiser:
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
        if restore_rng_state:
            restore_rng(state["rng"])
            gen.set_state(state["rng"]["data_order"])
        losses = train_epochs(model, opt, sched, gen, Xtr, ytr, EPOCHS - state["epoch"])
        return model, losses

    model_full, loss_resumed = resume(True, True)
    model_w, loss_weights = resume(False, False)
    model_nr, loss_norng = resume(True, False)

    second = np.array(loss_full[HALF * steps:])
    first_match = np.max(np.abs(np.array(loss_first) - np.array(loss_full[:HALF * steps])))
    rows = [
        ("uninterrupted", second, model_uninterrupted),
        ("full resume", np.array(loss_resumed), model_full),
        ("weights only", np.array(loss_weights), model_w),
        ("no RNG restore", np.array(loss_norng), model_nr),
    ]
    print(f"Model: {count_params(model_uninterrupted):,} parameters, {steps} steps per epoch, "
          f"checkpoint file {ckpt.stat().st_size / 1024:.0f} KB "
          f"(weights only {(results_dir() / 't14_weights_only.pt').stat().st_size / 1024:.0f} KB)")
    print(f"First halves identical: max |difference| = {first_match:.1e}\n")
    print(f"{'run':16s} {'loss, 20 steps':>15s} {'final loss':>11s} {'max |diff|':>11s} {'test acc':>9s}")
    print(f"{'':16s} {'after resume':>15s} {'(last 50)':>11s} {'vs uninterr.':>11s}")
    for name, l, m in rows:
        m.eval()
        print(f"{name:16s} {l[:20].mean():15.4f} {l[-50:].mean():11.4f} "
              f"{np.max(np.abs(l - second)):11.2e} {accuracy(m, Xte, yte):9.3f}")

    # 3. The BatchNorm and dropout trap: evaluating while still in training mode.
    m = model_uninterrupted
    m.eval()
    acc_eval = accuracy(m, Xte, yte)
    bn = m[2]
    saved_mean = bn.running_mean.clone()
    m.train()                          # forgot to call model.eval()
    acc_train_256 = accuracy(m, Xte, yte, batch=256)
    acc_train_8 = accuracy(m, Xte, yte, batch=8)
    drift = (bn.running_mean - saved_mean).norm().item() / saved_mean.norm().item()
    with torch.no_grad():
        p1 = m(Xte[:256]).argmax(1)
        p2 = m(Xte[:256]).argmax(1)
    flips = (p1 != p2).float().mean().item()
    print("\nEvaluating the uninterrupted model on the test set:")
    print(f"  model.eval()                       accuracy {acc_eval:.3f}")
    print(f"  forgot eval(), batches of 256      accuracy {acc_train_256:.3f}")
    print(f"  forgot eval(), batches of 8        accuracy {acc_train_8:.3f}")
    print(f"  same 256 images twice in train mode: {100 * flips:.1f}% of predictions changed (dropout)")
    print(f"  BatchNorm running mean moved by {100 * drift:.1f}% just by evaluating in train mode")

    # Figure: faint raw losses, solid moving averages, log scale so late differences show.
    def smooth(v, k=9):
        v = np.asarray(v, dtype=float)
        return np.convolve(np.pad(v, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [2.2, 1]})
    x0 = HALF * steps
    xs = np.arange(len(loss_full))
    curves = [("no RNG restore", loss_norng, "#8172B2", "-", 1.2),
              ("weights only", loss_weights, "#E07B39", "-", 1.6),
              ("uninterrupted", second, "#3B6EA8", "-", 2.2),
              ("full resume", loss_resumed, "#C44E52", (0, (2, 2)), 1.6)]
    a1.plot(xs[:x0], loss_full[:x0], color="#7F7F7F", lw=0.6, alpha=0.35)
    a1.plot(xs[:x0], smooth(loss_full[:x0]), color="#7F7F7F", lw=1.6, label="first 4 epochs (all runs)")
    for name, l, c, ls, lw in curves:
        a1.plot(xs[x0:], l, color=c, lw=0.6, alpha=0.25)
        a1.plot(xs[x0:], smooth(l), color=c, lw=lw, ls=ls, label=name)
    a1.axvline(x0, color="k", lw=0.8, ls=":")
    a1.set_yscale("log")
    a1.text(x0 + 4, 1.6, "save, stop,\nrestart", fontsize=8.5)
    a1.set_xlabel("training step")
    a1.set_ylabel("training loss (log scale)")
    a1.legend(fontsize=8, frameon=False, loc="lower left")
    a1.set_title("Resuming after a restart", fontsize=10, pad=8)
    names = ["model.eval()", "train mode,\nbatch 256", "train mode,\nbatch 8"]
    vals = [acc_eval, acc_train_256, acc_train_8]
    bars = a2.bar(names, vals, color=["#4C9A5F", "#E07B39", "#C44E52"], width=0.6)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v - 0.04, f"{v:.3f}", ha="center", va="top",
                fontsize=8.5, color="white", fontweight="bold")
    a2.set_ylim(max(0.0, min(vals) - 0.15), 1.0)
    a2.set_ylabel("test accuracy")
    a2.set_title("Forgetting model.eval()", fontsize=10, pad=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8.5)
    fig.tight_layout()
    out = results_dir() / "t14_checkpoint.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out.name}. Run time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
