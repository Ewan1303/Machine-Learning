"""Task 4 solution: backpropagation written by hand, checked two ways.

    python task_04_backprop.py

An MLP with ReLU hidden layers and a softmax cross-entropy loss. The backward pass uses
the equations derived in 1.3:

    dZ_L = (P - Y) / N                       output layer (softmax + cross-entropy)
    dW_l = dZ_l^T A_{l-1},  db_l = sum of dZ_l over the batch
    dA_{l-1} = dZ_l W_l,    dZ_{l-1} = dA_{l-1} * relu'(Z_{l-1})

The gradients are compared with central finite differences and with PyTorch autograd.
Everything is float64 so the checks are precise.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import make_spirals


def init(sizes, rng):
    return [[rng.normal(0, np.sqrt(2.0 / a), (b, a)), np.zeros(b)] for a, b in zip(sizes[:-1], sizes[1:])]


def forward(X, params):
    """Returns the logits and a cache of what the backward pass needs."""
    cache = [(None, X)]                      # (Z, A) per layer; layer 0 is the input
    A = X
    for k, (W, b) in enumerate(params):
        Z = A @ W.T + b
        A = np.maximum(0, Z) if k < len(params) - 1 else Z
        cache.append((Z, A))
    return A, cache


def loss_and_probs(logits, y):
    Zs = logits - logits.max(1, keepdims=True)
    P = np.exp(Zs) / np.exp(Zs).sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(len(y)), y]))
    return loss, P


def backward(params, cache, P, y):
    N = len(y)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0
    dZ = (P - Y) / N                          # gradient of the mean loss w.r.t. the logits
    grads = [None] * len(params)
    for k in reversed(range(len(params))):
        W, b = params[k]
        A_prev = cache[k][1]
        grads[k] = [dZ.T @ A_prev, dZ.sum(0)]  # dW has W's shape (n_out, n_in)
        if k > 0:
            dA = dZ @ W                         # pass the error back through W
            Z_prev = cache[k][0]
            dZ = dA * (Z_prev > 0)              # ReLU passes gradient only where it was active
    return grads


def numerical_grads(params, X, y, eps=1e-6):
    num = []
    for W, b in params:
        gW, gb = np.zeros_like(W), np.zeros_like(b)
        for arr, g in ((W, gW), (b, gb)):
            it = np.nditer(arr, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = arr[i]
                arr[i] = old + eps
                lp, _ = loss_and_probs(forward(X, params)[0], y)
                arr[i] = old - eps
                lm, _ = loss_and_probs(forward(X, params)[0], y)
                arr[i] = old
                g[i] = (lp - lm) / (2 * eps)   # central difference
        num.append([gW, gb])
    return num


def rel_error(a, b):
    return np.abs(a - b).max() / max(1e-12, np.abs(a).max() + np.abs(b).max())


rng = np.random.default_rng(1)
X, y = make_spirals(n=60, arms=3, seed=1)
X = X.astype(np.float64)
params = init([2, 8, 8, 3], rng)

logits, cache = forward(X, params)
loss, P = loss_and_probs(logits, y)
grads = backward(params, cache, P, y)
num = numerical_grads(params, X, y)

# PyTorch autograd on the same weights
tparams = [[torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)] for W, b in params]
A = torch.tensor(X)
for k, (W, b) in enumerate(tparams):
    A = A @ W.T + b
    if k < len(tparams) - 1:
        A = torch.relu(A)
tloss = torch.nn.functional.cross_entropy(A, torch.tensor(y))
tloss.backward()

print(f"loss: NumPy {loss:.10f} | PyTorch {tloss.item():.10f}")
print(f"{'parameter':<10}{'shape':>10}{'vs finite diff':>18}{'vs autograd':>15}")
for k, ((gW, gb), (nW, nb), (tW, tb)) in enumerate(zip(grads, num, tparams), 1):
    print(f"{'W' + str(k):<10}{str(gW.shape):>10}{rel_error(gW, nW):>18.2e}{rel_error(gW, tW.grad.numpy()):>15.2e}")
    print(f"{'b' + str(k):<10}{str(gb.shape):>10}{rel_error(gb, nb):>18.2e}{rel_error(gb, tb.grad.numpy()):>15.2e}")
print("A relative error below about 1e-7 means the backward pass is right.")

# What a bug looks like: forget the ReLU derivative in the backward pass
def buggy_backward(params, cache, P, y):
    N = len(y)
    Y = np.zeros_like(P); Y[np.arange(N), y] = 1.0
    dZ = (P - Y) / N
    out = [None] * len(params)
    for k in reversed(range(len(params))):
        W, b = params[k]
        out[k] = [dZ.T @ cache[k][1], dZ.sum(0)]
        if k > 0:
            dZ = dZ @ W                        # bug: no "* (Z_prev > 0)"
    return out

bug = buggy_backward(params, cache, P, y)
print(f"with the ReLU derivative left out, W1 error vs finite differences: {rel_error(bug[0][0], num[0][0]):.2e}")
