"""Task 6 solution: the same network in PyTorch, checked against the NumPy version.

    python task_06_pytorch.py

1. Copies the NumPy network's starting weights into an nn.Sequential and checks that the
   outputs and gradients match.
2. Trains the PyTorch model with the same optimiser settings as Task 5, then with Adam.
Saves results/task_06_pytorch_spirals.png.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from mlguide_data import make_spirals, train_val_test_split, plot_decision_boundary, results_dir, set_seed, count_params
from task_05_train_numpy import NumpyMLP


def make_model():
    return nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))


X, y = make_spirals(n=1500, arms=3, noise=0.2, seed=0)
(Xtr, ytr), (Xva, yva), (Xte, yte) = train_val_test_split(X, y, seed=0)

# ---- 1. Same weights in, same numbers out --------------------------------------------
np_model = NumpyMLP([2, 64, 64, 3], seed=0)
torch_model = make_model().double()
linears = [m for m in torch_model if isinstance(m, nn.Linear)]
with torch.no_grad():
    for lin, (W, b) in zip(linears, np_model.params):
        lin.weight.copy_(torch.from_numpy(W))
        lin.bias.copy_(torch.from_numpy(b))

Xb, yb = Xtr[:64].astype(np.float64), ytr[:64]
np_loss, np_grads = np_model.loss_and_grads(Xb, yb)
t_loss = nn.functional.cross_entropy(torch_model(torch.from_numpy(Xb)), torch.from_numpy(yb))
t_loss.backward()
max_diff = max(float(np.abs(g[0] - lin.weight.grad.numpy()).max()) for g, lin in zip(np_grads, linears))
print(f"loss  NumPy {np_loss:.12f}   PyTorch {t_loss.item():.12f}")
print(f"largest weight-gradient difference: {max_diff:.2e}")
print(f"parameters: {count_params(torch_model)} (the NumPy network has "
      f"{sum(W.size + b.size for W, b in np_model.params)})")


# ---- 2. Train in PyTorch -----------------------------------------------------------
def train(opt_name, epochs=300, batch=64, seed=0):
    set_seed(seed)
    model = make_model()
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[200], gamma=0.2)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()                       # clear old gradients
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()                       # backprop fills .grad
            opt.step()                            # update the weights
        sched.step()
    model.eval()
    with torch.no_grad():
        acc = lambda Xa, ya: (model(torch.from_numpy(Xa)).argmax(1).numpy() == ya).mean()
        res = dict(train=acc(Xtr, ytr), val=acc(Xva, yva), test=acc(Xte, yte), seconds=time.time() - t0)
    return model, res


print(f"\n{'optimiser':<22}{'train acc':>10}{'val acc':>10}{'test acc':>10}{'time (s)':>10}")
models = {}
for name, label in (("sgd", "SGD + momentum 0.9"), ("adam", "Adam, cosine LR")):
    m, r = train(name)
    models[name] = m
    print(f"{label:<22}{r['train']:>10.3f}{r['val']:>10.3f}{r['test']:>10.3f}{r['seconds']:>10.1f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, title) in zip(axes, (("sgd", "PyTorch, SGD + momentum"), ("adam", "PyTorch, Adam"))):
    m = models[name]
    plot_decision_boundary(lambda G: m(torch.from_numpy(G)).argmax(1).detach().numpy(), Xtr, ytr, ax, title=title)
fig.tight_layout()
out = results_dir() / "task_06_pytorch_spirals.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
