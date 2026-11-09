"""Task 7 solution: check the parameter counts, compute and memory of three designs.

    python task_07_design.py

The three problems from 5.1:
  (a) tabular data, 20 numeric features, 10,000 rows, binary label
  (b) 28 x 28 greyscale images, 10 classes, 60,000 images
  (c) sequences of 100 steps with 8 features each, 5 classes, 20,000 sequences
For each design the script compares the hand formula with PyTorch's count, estimates the
forward-pass FLOPs per example and the memory needed to train with Adam in float32.
"""
import torch
import torch.nn as nn


def count(model):
    return sum(p.numel() for p in model.parameters())


# (a) Tabular: an MLP 20 -> 128 -> 64 -> 1
mlp = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
mlp_formula = (20 * 128 + 128) + (128 * 64 + 64) + (64 * 1 + 1)
mlp_flops = 2 * (20 * 128 + 128 * 64 + 64 * 1)               # 2 FLOPs (multiply + add) per weight


# (b) Images: a small CNN, 3x3 convolutions, channels 32-32-64-64, global average pooling
def conv_block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]


cnn = nn.Sequential(*conv_block(1, 32), *conv_block(32, 32), nn.MaxPool2d(2),
                    *conv_block(32, 64), *conv_block(64, 64), nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
conv = lambda k, ci, co: k * k * ci * co + co                 # weights + biases of a conv layer
bn = lambda c: 2 * c                                          # gamma and beta (running stats are buffers)
cnn_formula = (conv(3, 1, 32) + bn(32) + conv(3, 32, 32) + bn(32) + conv(3, 32, 64) + bn(64)
               + conv(3, 64, 64) + bn(64) + (64 * 10 + 10))
cnn_flops = 2 * (9 * 1 * 32 * 28 * 28 + 9 * 32 * 32 * 28 * 28 + 9 * 32 * 64 * 14 * 14
                 + 9 * 64 * 64 * 14 * 14 + 64 * 10)


# (c) Sequences: a one-layer GRU with 64 hidden units, then a linear head on the last state
class GRUClassifier(nn.Module):
    def __init__(self, n_in=8, hidden=64, n_out=5):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                    # x: (batch, time, features)
        out, h = self.gru(x)                 # h: (1, batch, hidden), the final hidden state
        return self.head(h[-1])


gru = GRUClassifier()
h, x = 64, 8
gru_formula = 3 * (h * x + h * h + 2 * h) + (h * 5 + 5)       # PyTorch keeps two bias vectors per gate
gru_flops = 100 * 2 * 3 * (h * x + h * h) + 2 * h * 5

rows = [
    ("(a) MLP 20-128-64-1", mlp, mlp_formula, mlp_flops, torch.randn(32, 20), 10_000),
    ("(b) CNN 32-32-64-64", cnn, cnn_formula, cnn_flops, torch.randn(32, 1, 28, 28), 60_000),
    ("(c) GRU 64 + head", gru, gru_formula, gru_flops, torch.randn(32, 100, 8), 20_000),
]
print(f"{'design':<22}{'formula':>9}{'PyTorch':>9}{'output shape':>16}{'MFLOPs/ex':>11}{'Adam MB':>9}{'examples/param':>16}")
for name, model, formula, flops, dummy, n_data in rows:
    n = count(model)
    out_shape = tuple(model(dummy).shape)
    train_mb = n * 16 / 1e6        # fp32 weights + grads + two Adam moments = 16 bytes per parameter
    print(f"{name:<22}{formula:>9,}{n:>9,}{str(out_shape):>16}{flops / 1e6:>11.2f}{train_mb:>9.2f}{n_data / n:>16.2f}")

print("\nAll three fit easily on a laptop: the CNN costs most compute per example because")
print("each 3x3 kernel is reused at every one of the 784 (or 196) positions.")
