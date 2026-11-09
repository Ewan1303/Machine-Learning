"""Task 10 solution: a training loop that saves and resumes exactly.

    python task_10_checkpoint.py

Compares three runs of the same 6-epoch training job:
  A  uninterrupted
  B  stopped after 4 epochs, full checkpoint saved, everything rebuilt, checkpoint loaded, finished
  C  stopped after 4 epochs, only the weights saved (fresh optimiser, scheduler and random state)
A full checkpoint makes B identical to A, bit for bit. C drifts off and its loss jumps at the
restart. Finally it shows the BatchNorm/Dropout eval-mode trap.
Saves results/task_10_checkpoint.png.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import synthetic_digits, train_val_test_split, set_seed, results_dir

CKPT_DIR = results_dir() / "task_10_checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
EPOCHS, BATCH, STOP = 6, 64, 4      # the job is interrupted after STOP epochs

X, y = synthetic_digits(n_per_class=300, seed=0)
X = X.reshape(len(X), -1)                                   # 16 x 16 images -> 256 inputs
(Xtr, ytr), (Xva, yva), _ = train_val_test_split(X, y, val=0.2, test=0.0, seed=0)
Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
STEPS_PER_EPOCH = (len(Xtr) + BATCH - 1) // BATCH


def build(seed):
    """Everything a run needs: model, optimiser, LR schedule and the data-order generator."""
    set_seed(seed)
    model = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * STEPS_PER_EPOCH)
    gen = torch.Generator().manual_seed(seed)              # controls the shuffling order
    return dict(model=model, opt=opt, sched=sched, gen=gen, epoch=0, step=0, losses=[])


def train(run, n_epochs):
    model, opt, sched, gen = run["model"], run["opt"], run["sched"], run["gen"]
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            sched.step()
            run["step"] += 1
            run["losses"].append(loss.item())
        run["epoch"] += 1


def save_checkpoint(run, path):
    np_state = np.random.get_state()
    ckpt = {
        "model": run["model"].state_dict(),         # weights AND buffers (BatchNorm running stats)
        "optimizer": run["opt"].state_dict(),       # Adam's moment estimates and step counts
        "scheduler": run["sched"].state_dict(),     # where we are in the LR schedule
        "generator": run["gen"].get_state(),        # data order
        "torch_rng": torch.get_rng_state(),         # dropout masks
        "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],   # lists keep weights_only=True loads working
        "python_rng": random.getstate(),
        "epoch": run["epoch"], "step": run["step"], "losses": run["losses"],
        "config": {"epochs": EPOCHS, "batch": BATCH, "lr": 2e-3},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)            # atomic: a crash mid-save never leaves a half-written file


def load_checkpoint(run, path):
    ckpt = torch.load(path, weights_only=True)      # safe loading: tensors and plain Python only
    run["model"].load_state_dict(ckpt["model"])
    run["opt"].load_state_dict(ckpt["optimizer"])
    run["sched"].load_state_dict(ckpt["scheduler"])
    run["gen"].set_state(ckpt["generator"])
    torch.set_rng_state(ckpt["torch_rng"])
    s = ckpt["numpy_rng"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), *s[2:]))
    random.setstate(ckpt["python_rng"])
    run["epoch"], run["step"], run["losses"] = ckpt["epoch"], ckpt["step"], list(ckpt["losses"])


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.state_dict().values() if p.dtype.is_floating_point])


def val_acc(model, train_mode=False):
    model.train(train_mode)
    with torch.no_grad():
        return (model(Xva).argmax(1) == yva).float().mean().item()


# A: uninterrupted
A = build(seed=0)
train(A, EPOCHS)

# B: stop, save everything, throw it all away, rebuild (even with a different seed), load, finish
B = build(seed=0)
train(B, STOP)
save_checkpoint(B, CKPT_DIR / "full.pt")
del B
B = build(seed=123)
load_checkpoint(B, CKPT_DIR / "full.pt")
train(B, EPOCHS - STOP)

# C: stop, save only the weights, rebuild everything else fresh
C = build(seed=0)
train(C, STOP)
torch.save(C["model"].state_dict(), CKPT_DIR / "weights_only.pt")
C_losses = list(C["losses"])
C = build(seed=1)
C["model"].load_state_dict(torch.load(CKPT_DIR / "weights_only.pt", weights_only=True))
C["losses"] = C_losses
lr_at_restart_A = 2e-3 * 0.5 * (1 + np.cos(np.pi * STOP / EPOCHS))
lr_at_restart_C = C["opt"].param_groups[0]["lr"]
train(C, EPOCHS - STOP)    # the scheduler restarts at the top LR and Adam's moments start at zero

ref = flat_params(A["model"])
print(f"{'run':<34}{'final loss':>11}{'max |weight diff| vs A':>24}{'val acc':>9}")
for name, run in (("A uninterrupted", A), ("B full checkpoint, resumed", B), ("C weights only, resumed", C)):
    diff = (flat_params(run["model"]) - ref).abs().max().item()
    print(f"{name:<34}{np.mean(run['losses'][-STEPS_PER_EPOCH:]):>11.4f}{diff:>24.3e}{val_acc(run['model']):>9.3f}")

k = STOP * STEPS_PER_EPOCH
print(f"\nlearning rate at the restart: A and B {lr_at_restart_A:.1e}, C {lr_at_restart_C:.1e}")
print(f"mean loss over the 10 batches straight after the restart (step {k}):")
for name, run in (("A", A), ("B", B), ("C", C)):
    print(f"  {name}: {np.mean(run['losses'][k:k + 10]):.4f}")
print(f"checkpoint file sizes: full {os.path.getsize(CKPT_DIR / 'full.pt') / 1e3:.0f} kB, "
      f"weights only {os.path.getsize(CKPT_DIR / 'weights_only.pt') / 1e3:.0f} kB")

# The eval-mode trap: BatchNorm and Dropout behave differently in train mode
print(f"\nvalidation accuracy of A with model.eval(): {val_acc(A['model'], False):.3f}")
print(f"validation accuracy of A left in train mode: {val_acc(A['model'], True):.3f}")
try:
    A["model"].train()
    A["model"](Xva[:1])
except ValueError as ex:
    print("one example in train mode fails:", str(ex).split(",")[0])

fig, ax = plt.subplots(figsize=(7, 3.2))
smooth = lambda v: np.convolve(v, np.ones(5) / 5, mode="valid")
ax.plot(smooth(A["losses"]), color="#3B6EA8", lw=2.5, label="A uninterrupted")
ax.plot(smooth(B["losses"]), color="#E07B39", lw=1.2, ls="--", label="B full checkpoint (on top of A)")
ax.plot(smooth(C["losses"]), color="#C44E52", lw=1.2, label="C weights only")
ax.axvline(k, color="#7F7F7F", ls=":", lw=1)
ax.text(k + 3, ax.get_ylim()[1] * 0.9, "restart", color="#7F7F7F", fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("batch loss (5-step mean)")
ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(results_dir() / "task_10_checkpoint.png", dpi=150, bbox_inches="tight")
print("saved", results_dir() / "task_10_checkpoint.png")
