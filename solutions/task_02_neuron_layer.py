"""Task 2 solution: a neuron and a layer in NumPy, checked against PyTorch.

    python task_02_neuron_layer.py

Shapes follow PyTorch: a batch X is (N, n_in), a weight matrix W is (n_out, n_in),
a bias b is (n_out,), and a layer computes act(X @ W.T + b) with shape (N, n_out).
"""
import numpy as np
import torch


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def neuron(x, w, b, act=relu):
    """One neuron: x and w are vectors of length n_in, b is a number."""
    return act(np.dot(w, x) + b)


def layer(X, W, b, act=relu):
    """A layer of n_out neurons applied to a batch X of shape (N, n_in)."""
    assert X.ndim == 2 and W.shape[1] == X.shape[1] and b.shape == (W.shape[0],)
    Z = X @ W.T + b          # (N, n_in) @ (n_in, n_out) -> (N, n_out); b broadcasts over rows
    return act(Z)


rng = np.random.default_rng(0)
N, n_in, n_out = 5, 3, 4
X = rng.normal(size=(N, n_in))
W = rng.normal(size=(n_out, n_in)) * 0.5
b = rng.normal(size=n_out) * 0.1

# 1. The layer equals running each neuron on each example separately
H = layer(X, W, b)
H_loop = np.array([[neuron(X[i], W[j], b[j]) for j in range(n_out)] for i in range(N)])
print("layer output shape:", H.shape)
print("max |layer - loop over neurons| =", np.abs(H - H_loop).max())

# 2. Broadcasting the bias: Z = X W^T + b adds b to every row
Z = X @ W.T
print("adding b by broadcasting equals adding it row by row:",
      np.allclose(Z + b, Z + np.tile(b, (N, 1))))

# 3. Compare with torch.nn.Linear given the same weights
lin = torch.nn.Linear(n_in, n_out)
with torch.no_grad():
    lin.weight.copy_(torch.from_numpy(W))
    lin.bias.copy_(torch.from_numpy(b))
H_torch = torch.relu(lin(torch.from_numpy(X).float())).detach().numpy()
print("max |NumPy - PyTorch| =", float(np.abs(H - H_torch).max()))

# 4. Parameter count of the layer
print("parameters: n_in*n_out + n_out =", n_in * n_out + n_out,
      "| PyTorch says", sum(p.numel() for p in lin.parameters()))

# 5. A sigmoid layer gives values in (0, 1); ReLU zeroes the negatives
print("sigmoid layer range:", layer(X, W, b, sigmoid).min().round(3), "to", layer(X, W, b, sigmoid).max().round(3))
print("fraction of ReLU outputs that are exactly 0:", (H == 0).mean().round(3))
