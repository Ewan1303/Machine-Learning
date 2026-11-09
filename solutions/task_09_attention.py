"""Task 9 solution: scaled dot-product and multi-head attention from scratch.

    python task_09_attention.py

Checks the hand-written versions against F.scaled_dot_product_attention and
nn.MultiheadAttention, then shows why the scores are divided by sqrt(d_k).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def attention(Q, K, V, causal=False):
    """Q: (..., n_q, d_k), K: (..., n_k, d_k), V: (..., n_k, d_v)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)          # (..., n_q, n_k)
    if causal:                                                 # position t may only look at 0..t
        n_q, n_k = scores.shape[-2:]
        mask = torch.ones(n_q, n_k, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)                           # each row sums to 1
    return weights @ V, weights


B, n, d = 2, 6, 16
Q, K, V = torch.randn(B, n, d), torch.randn(B, n, d), torch.randn(B, n, d)
for causal in (False, True):
    mine, w = attention(Q, K, V, causal)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    print(f"single head, causal={causal}: max difference {(mine - ref).abs().max().item():.2e}")
print("causal weights for the first sequence (rows = queries):")
print(w[0].numpy().round(2))


class MyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_head = n_heads, d_model // n_heads
        self.in_proj = nn.Linear(d_model, 3 * d_model)    # Q, K and V projections stacked
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):                                  # x: (B, n, d_model)
        B, n, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        split = lambda t: t.view(B, n, self.h, self.d_head).transpose(1, 2)   # (B, h, n, d_head)
        out, _ = attention(split(q), split(k), split(v))
        out = out.transpose(1, 2).reshape(B, n, self.h * self.d_head)        # concatenate heads
        return self.out_proj(out)


d_model, heads = 32, 4
ref_mha = nn.MultiheadAttention(d_model, heads, batch_first=True)
my_mha = MyMultiHeadAttention(d_model, heads)
with torch.no_grad():
    my_mha.in_proj.weight.copy_(ref_mha.in_proj_weight)
    my_mha.in_proj.bias.copy_(ref_mha.in_proj_bias)
    my_mha.out_proj.weight.copy_(ref_mha.out_proj.weight)
    my_mha.out_proj.bias.copy_(ref_mha.out_proj.bias)
x = torch.randn(B, n, d_model)
ref_out, _ = ref_mha(x, x, x, need_weights=False)
print(f"\nmulti-head ({heads} heads): max difference {(my_mha(x) - ref_out).abs().max().item():.2e}")
n_par = sum(p.numel() for p in ref_mha.parameters())
print(f"parameters: 4d^2 + 4d = {4 * d_model ** 2 + 4 * d_model}, PyTorch = {n_par}")

# Why divide by sqrt(d_k)? Dot products of random unit-variance vectors have variance d_k.
print(f"\n{'d_k':>5}{'std of q.k':>12}{'std scaled':>12}{'max weight, raw':>17}{'max weight, scaled':>20}")
for d_k in (4, 16, 64, 256, 1024):
    q, k = torch.randn(2000, d_k), torch.randn(2000, 10, d_k)
    raw = (k @ q.unsqueeze(-1)).squeeze(-1)              # 2000 queries, 10 keys each
    scaled = raw / math.sqrt(d_k)
    print(f"{d_k:>5}{raw.std().item():>12.2f}{scaled.std().item():>12.2f}"
          f"{raw.softmax(-1).max(-1).values.mean().item():>17.3f}{scaled.softmax(-1).max(-1).values.mean().item():>20.3f}")
print("Without scaling, larger d_k pushes the softmax towards one-hot weights, where its gradient is tiny.")
