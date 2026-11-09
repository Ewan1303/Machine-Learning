"""Small datasets and helpers shared by every test and task in the Machine Learning guide.

Everything here is generated on your own machine in a few seconds, so nothing has to be
downloaded. Needs numpy, torch and matplotlib only.

    from mlguide_data import make_spirals, synthetic_digits, set_seed, count_params

Datasets
    make_xor, make_moons, make_circles, make_spirals, make_blobs   2-D toy classification
    make_sine                                                     1-D regression
    synthetic_digits                                              16x16 images of the digits 0-9
    permute_pixels, split_by_class, train_val_test_split           helpers for building tasks

Helpers
    set_seed, count_params, plot_decision_boundary, results_dir
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- helpers
def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def count_params(model, trainable_only: bool = True) -> int:
    """Number of parameters (weights and biases) in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def results_dir() -> Path:
    """Folder the tests save their figures into (created if missing)."""
    d = HERE / "results"
    d.mkdir(exist_ok=True)
    return d


def train_val_test_split(X, y, val=0.15, test=0.15, seed=0):
    """Shuffle and split arrays into train, validation and test parts."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test, n_val = int(len(X) * test), int(len(X) * val)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


# --------------------------------------------------------------------------- 2-D toy data
def make_xor(n=400, noise=0.15, seed=0):
    """Four clusters at the corners of a square; opposite corners share a class (XOR)."""
    rng = np.random.default_rng(seed)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0])
    k = rng.integers(0, 4, n)
    X = corners[k] + rng.normal(0, noise, (n, 2))
    return X.astype(np.float32), labels[k].astype(np.int64)


def make_moons(n=400, noise=0.1, seed=0):
    """Two interleaving half circles."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t0 = rng.uniform(0, np.pi, n0)
    t1 = rng.uniform(0, np.pi, n - n0)
    a = np.stack([np.cos(t0), np.sin(t0)], 1)
    b = np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], 1)
    X = np.concatenate([a, b]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n0), np.ones(n - n0)])
    p = rng.permutation(n)
    return X[p].astype(np.float32), y[p].astype(np.int64)


def make_circles(n=400, noise=0.08, factor=0.5, seed=0):
    """A small circle (class 1) inside a large one (class 0)."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.where(np.arange(n) < n0, 1.0, factor)
    X = np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))
    y = (np.arange(n) >= n0).astype(np.int64)
    p = rng.permutation(n)
    return X[p].astype(np.float32), y[p]


def make_spirals(n=600, arms=2, noise=0.2, turns=1.5, seed=0):
    """Interleaved spirals, one per class. Hard for small or linear models."""
    rng = np.random.default_rng(seed)
    per = n // arms
    Xs, ys = [], []
    for c in range(arms):
        t = np.sqrt(rng.uniform(0, 1, per)) * turns * 2 * np.pi
        r = t / (turns * 2 * np.pi) * 3
        ang = t + 2 * np.pi * c / arms
        pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + rng.normal(0, noise * 0.5, (per, 2))
        Xs.append(pts)
        ys.append(np.full(per, c))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    p = rng.permutation(len(X))
    return X[p].astype(np.float32), y[p].astype(np.int64)


def make_blobs(n=400, centers=((-2, -2), (2, 2)), std=0.7, seed=0):
    """Gaussian clusters, one class per centre."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=np.float32)
    k = rng.integers(0, len(centers), n)
    X = centers[k] + rng.normal(0, std, (n, centers.shape[1]))
    return X.astype(np.float32), k.astype(np.int64)


def make_sine(n=200, noise=0.1, x_range=(-3, 3), seed=0):
    """y = sin(x) plus noise, for 1-D regression. Returns X with shape (n, 1) and y with shape (n, 1)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)
    y = np.sin(x) + rng.normal(0, noise, n)
    return x[:, None].astype(np.float32), y[:, None].astype(np.float32)


# --------------------------------------------------------------------------- synthetic digits
_FONTS = [("DejaVu Sans", "normal"), ("DejaVu Sans", "bold"), ("DejaVu Serif", "normal"),
          ("DejaVu Serif", "bold"), ("DejaVu Sans Mono", "normal"), ("DejaVu Sans Mono", "bold"),
          ("STIXGeneral", "normal"), ("STIXGeneral", "bold")]


def _glyphs(res=64):
    """Render each digit in each font once, centred, as res x res greyscale arrays in [0, 1]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = np.zeros((10, len(_FONTS), res, res), dtype=np.float32)
    for s, (family, weight) in enumerate(_FONTS):
        for d in range(10):
            fig = plt.figure(figsize=(1, 1), dpi=res)
            fig.text(0.5, 0.5, str(d), fontsize=res * 0.62, ha="center", va="center",
                     family=family, weight=weight)
            fig.canvas.draw()
            img = 1.0 - np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float32) / 255.0
            plt.close(fig)
            # centre by bounding box and scale so the glyph is about 70% of the height
            ys, xs = np.nonzero(img > 0.2)
            if len(ys) == 0:
                continue
            crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape
            scale = 0.7 * res / h
            from PIL import Image
            im = Image.fromarray((crop * 255).astype(np.uint8)).resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
            c = np.asarray(im).astype(np.float32) / 255.0
            canvas = np.zeros((res, res), dtype=np.float32)
            y0, x0 = (res - c.shape[0]) // 2, (res - c.shape[1]) // 2
            canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c[: res - y0, : res - x0]
            out[d, s] = canvas
    return out


def synthetic_digits(n_per_class=600, size=16, seed=0, max_shift=1.0, max_rotate=12.0,
                     scale_range=(0.85, 1.1), noise=0.05, classes=range(10), cache=True):
    """Images of the digits 0-9 drawn in eight fonts with random distortions.

    Returns X with shape (N, 1, size, size), values in [0, 1], and integer labels y.
    max_shift is the largest random shift in pixels of the final image, max_rotate in degrees.
    Results are cached in a data_cache folder next to this file, so later calls are instant.
    """
    import torch
    import torch.nn.functional as F
    classes = list(classes)
    key = f"digits_n{n_per_class}_s{size}_seed{seed}_sh{max_shift}_r{max_rotate}_sc{scale_range[0]}-{scale_range[1]}_no{noise}_c{''.join(map(str, classes))}"
    cdir = HERE / "data_cache"
    cfile = cdir / f"{key}.npz"
    if cache and cfile.exists():
        z = np.load(cfile)
        return z["X"], z["y"]
    gfile = cdir / "glyphs64.npy"
    if gfile.exists():
        G = np.load(gfile)
    else:
        G = _glyphs(64)
        cdir.mkdir(exist_ok=True)
        np.save(gfile, G)
    rng = np.random.default_rng(seed)
    N = n_per_class * len(classes)
    y = np.repeat(np.array(classes), n_per_class)
    style = rng.integers(0, G.shape[1], N)
    base = torch.from_numpy(G[y, style]).unsqueeze(1)          # (N, 1, 64, 64)
    # random stroke thickness: thicken a third, thin a sixth
    r = rng.uniform(size=N)
    thick = torch.from_numpy(r < 1 / 3)
    thin = torch.from_numpy(r > 5 / 6)
    dil = F.max_pool2d(base, 3, 1, 1)
    ero = -F.max_pool2d(-base, 3, 1, 1)
    base = torch.where(thick[:, None, None, None], dil, torch.where(thin[:, None, None, None], ero, base))
    # random affine transform
    ang = np.deg2rad(rng.uniform(-max_rotate, max_rotate, N))
    sc = rng.uniform(*scale_range, N)
    sh = rng.uniform(-0.15, 0.15, N)
    tx = rng.uniform(-max_shift, max_shift, N) * 2 / size
    ty = rng.uniform(-max_shift, max_shift, N) * 2 / size
    cos, sin = np.cos(ang), np.sin(ang)
    A = np.zeros((N, 2, 3), dtype=np.float32)
    A[:, 0, 0] = cos / sc
    A[:, 0, 1] = (-sin + sh) / sc
    A[:, 1, 0] = sin / sc
    A[:, 1, 1] = cos / sc
    A[:, 0, 2] = -tx
    A[:, 1, 2] = -ty
    grid = F.affine_grid(torch.from_numpy(A), base.shape, align_corners=False)
    warped = F.grid_sample(base, grid, align_corners=False, padding_mode="zeros")
    small = F.interpolate(warped, size=(size, size), mode="area")
    small = small / small.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    small = small + torch.from_numpy(rng.normal(0, noise, small.shape).astype(np.float32))
    X = small.clamp(0, 1).numpy().astype(np.float32)
    p = rng.permutation(N)
    X, y = X[p], y[p].astype(np.int64)
    if cache:
        cdir.mkdir(exist_ok=True)
        np.savez_compressed(cfile, X=X, y=y)
    return X, y


def permute_pixels(X, seed):
    """Apply one fixed random permutation of pixel positions to every image (a 'permuted' task)."""
    rng = np.random.default_rng(seed)
    flat = X.reshape(len(X), -1)
    perm = rng.permutation(flat.shape[1])
    return flat[:, perm].reshape(X.shape)


def split_by_class(X, y, classes, relabel=False):
    """Keep only the given classes. With relabel=True they become 0, 1, 2, ... in the given order."""
    classes = list(classes)
    mask = np.isin(y, classes)
    Xs, ys = X[mask], y[mask]
    if relabel:
        lut = {c: i for i, c in enumerate(classes)}
        ys = np.array([lut[v] for v in ys], dtype=np.int64)
    return Xs, ys


# --------------------------------------------------------------------------- plotting
def plot_decision_boundary(predict_fn, X, y, ax, title=None, resolution=300, pad=0.5):
    """Shade the regions a 2-D classifier assigns to each class and draw the data on top.

    predict_fn takes an (M, 2) float32 array and returns class indices (or probabilities of class 1).
    """
    import matplotlib.pyplot as plt  # noqa: F401
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    out = np.asarray(predict_fn(grid))
    if out.ndim == 2:
        out = out.argmax(1)
    zz = out.reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=np.arange(zz.max() + 2) - 0.5, cmap="coolwarm", alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


if __name__ == "__main__":
    # Quick self-test: build each dataset and save a preview image.
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = time.time()
    X, y = synthetic_digits(n_per_class=200, seed=1, cache=False)
    print(f"synthetic_digits: {X.shape}, {y.shape}, {time.time() - t:.1f}s")
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(X[k, 0], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(str(y[k]), fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(results_dir() / "preview_digits.png", dpi=110)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, (name, f) in zip(axes, [("xor", make_xor), ("moons", make_moons), ("circles", make_circles),
                                    ("spirals", make_spirals), ("blobs", make_blobs)]):
        Xd, yd = f()
        ax.scatter(Xd[:, 0], Xd[:, 1], c=yd, cmap="coolwarm", s=8); ax.set_title(name)
    fig.tight_layout(); fig.savefig(results_dir() / "preview_2d.png", dpi=90)
    print("saved previews to", results_dir())
