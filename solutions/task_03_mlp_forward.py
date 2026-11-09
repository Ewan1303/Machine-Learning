"""Task 3 solution: forward pass and parameter count of a small MLP in NumPy.

    python task_03_mlp_forward.py

Network: 2 inputs -> 16 ReLU -> 16 ReLU -> 3 outputs (softmax), on 3-arm spirals.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find mlguide_data.py
from mlguide_data import make_spirals

sizes = [2, 16, 16, 3]
rng = np.random.default_rng(0)

# He initialisation: weights ~ N(0, 2/n_in), biases 0
params = []
for n_in, n_out in zip(sizes[:-1], sizes[1:]):
    W = rng.normal(0, np.sqrt(2.0 / n_in), size=(n_out, n_in))
    b = np.zeros(n_out)
    params.append((W, b))


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # subtract the row max so exp never overflows
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(X, params):
    A = X
    shapes = [A.shape]
    for k, (W, b) in enumerate(params):
        Z = A @ W.T + b
        A = np.maximum(0, Z) if k < len(params) - 1 else Z   # ReLU on hidden layers, logits at the end
        shapes.append(A.shape)
    return A, shapes


X, y = make_spirals(n=600, arms=3, seed=0)
logits, shapes = forward(X, params)
P = softmax(logits)

# Parameter count by formula: sum over layers of n_in*n_out + n_out
by_formula = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
by_code = sum(W.size + b.size for W, b in params)
print("layer-by-layer shapes:", shapes)
print("parameters by formula:", by_formula, "| counted from the arrays:", by_code)
for k, (a, b_) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
    print(f"  layer {k}: {a} x {b_} weights + {b_} biases = {a * b_ + b_}")

print("each softmax row sums to 1:", np.allclose(P.sum(1), 1.0))
loss = -np.mean(np.log(P[np.arange(len(y)), y]))
print(f"cross-entropy at initialisation: {loss:.4f}  (ln 3 = {np.log(3):.4f})")
print(f"accuracy at initialisation: {(P.argmax(1) == y).mean():.3f}  (chance = 0.333)")

# Scaling the last layer's weights down makes the initial loss closer to ln 3
W3, b3 = params[-1]
small = params[:-1] + [(W3 * 0.01, b3)]
P_small = softmax(forward(X, small)[0])
loss_small = -np.mean(np.log(P_small[np.arange(len(y)), y]))
print(f"with the output weights scaled by 0.01: {loss_small:.4f}")
