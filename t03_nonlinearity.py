"""Test 3: why non-linearity matters.

Three models are trained on two datasets that no straight line can separate (XOR and
two interleaved spirals):

    1. a linear model                    (one layer, 2 -> 2)
    2. an MLP with no activation         (2 -> 32 -> 32 -> 2, layers stacked directly)
    3. the same MLP with ReLU            (2 -> 32 -> ReLU -> 32 -> ReLU -> 2)

Model 2 has 1,218 parameters but can still only draw a straight boundary: stacking
linear layers gives another linear function. The script proves this by multiplying
its three weight matrices into one and checking that the single matrix gives the
same outputs.

Run:
    python t03_nonlinearity.py

What to look for: models 1 and 2 behave the same and get stuck well short of 100%;
only model 3 bends its boundary around the data.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mlguide_data import (count_params, make_spirals, make_xor, plot_decision_boundary,
                          results_dir, set_seed)

MODELS = {
    "linear model": lambda: nn.Linear(2, 2),
    "MLP, no activation": lambda: nn.Sequential(nn.Linear(2, 32), nn.Linear(32, 32),
                                                nn.Linear(32, 2)),
    "MLP with ReLU": lambda: nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32),
                                           nn.ReLU(), nn.Linear(32, 2)),
}


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def main():
    t0 = time.time()
    data = {
        "XOR": (make_xor(n=400, noise=0.25, seed=0), make_xor(n=400, noise=0.25, seed=1)),
        "spirals": (make_spirals(n=600, noise=0.2, seed=0), make_spirals(n=600, noise=0.2, seed=1)),
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    print(f"{'dataset':8s} {'model':20s} {'params':>7s} {'train acc':>10s} {'test acc':>9s}")
    for r, (dname, ((Xtr, ytr), (Xte, yte))) in enumerate(data.items()):
        Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte)
        for c, (mname, make) in enumerate(MODELS.items()):
            set_seed(0)
            model = make()
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            for step in range(3000):                 # full-batch training
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(Xtr_t), ytr_t)
                loss.backward()
                opt.step()
            tr, te = accuracy(model, Xtr_t, ytr_t), accuracy(model, Xte_t, yte_t)
            print(f"{dname:8s} {mname:20s} {count_params(model):7d} {tr:10.3f} {te:9.3f}")
            pred = lambda G: model(torch.from_numpy(G)).argmax(1).numpy()
            with torch.no_grad():
                plot_decision_boundary(pred, Xte, yte, axes[r, c],
                                       title=f"{dname}: {mname}\ntest accuracy {te:.0%}")
            if mname == "MLP, no activation" and dname == "spirals":
                collapsed = model
    # ---- prove the no-activation MLP is one linear map ---------------------------------
    L1, L2, L3 = collapsed[0], collapsed[1], collapsed[2]
    with torch.no_grad():
        W = L3.weight @ L2.weight @ L1.weight                       # 2 x 2
        b = L3.weight @ (L2.weight @ L1.bias + L2.bias) + L3.bias  # 2
        Xte = torch.from_numpy(data["spirals"][1][0])
        diff = (collapsed(Xte) - (Xte @ W.T + b)).abs().max().item()
    print("\nThe 3-layer MLP without activations, multiplied out into one layer:")
    fmt = lambda a: np.array2string(a.numpy(), precision=3, suppress_small=True)
    print(f"W = {fmt(W)}")
    print(f"b = {fmt(b)}")
    print(f"largest difference between the 3-layer net and the single layer: {diff:.2e}")
    fig.tight_layout()
    out = results_dir() / "t03_nonlinearity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
