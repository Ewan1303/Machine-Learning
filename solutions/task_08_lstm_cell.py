"""Task 8 solution: an LSTM cell written from the equations, checked against nn.LSTM.

    python task_08_lstm_cell.py

PyTorch stacks the four gates in the order input (i), forget (f), cell candidate (g),
output (o), in weight_ih_l0 (4h x n_in), weight_hh_l0 (4h x h) and two bias vectors.

    i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
    f = sigmoid(W_if x + b_if + W_hf h + b_hf)
    g = tanh(W_ig x + b_ig + W_hg h + b_hg)
    o = sigmoid(W_io x + b_io + W_ho h + b_ho)
    c' = f * c + i * g
    h' = o * tanh(c')
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
n_in, hidden, T, batch = 5, 7, 12, 3


class MyLSTMCell(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.W_ih = nn.Parameter(torch.randn(4 * hidden, n_in) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden, hidden) * 0.1)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden))

    def forward(self, x, state):
        h, c = state
        gates = x @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh   # (batch, 4h)
        i, f, g, o = gates.chunk(4, dim=1)                                  # PyTorch's gate order
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g            # the cell state: keep part of the old, add part of the new
        h = o * torch.tanh(c)        # the output reads a squashed copy of the cell
        return h, c


ref = nn.LSTM(n_in, hidden, batch_first=True)
mine = MyLSTMCell(n_in, hidden)
with torch.no_grad():                # copy PyTorch's weights into the hand-written cell
    mine.W_ih.copy_(ref.weight_ih_l0)
    mine.W_hh.copy_(ref.weight_hh_l0)
    mine.b_ih.copy_(ref.bias_ih_l0)
    mine.b_hh.copy_(ref.bias_hh_l0)

x = torch.randn(batch, T, n_in)
out_ref, (h_ref, c_ref) = ref(x)

h = torch.zeros(batch, hidden)
c = torch.zeros(batch, hidden)
outs = []
for t in range(T):                   # unroll over time with the same weights at every step
    h, c = mine(x[:, t], (h, c))
    outs.append(h)
out_mine = torch.stack(outs, dim=1)

print("output shape:", tuple(out_mine.shape))
print(f"max |h difference| over all steps: {(out_mine - out_ref).abs().max().item():.2e}")
print(f"max |final c difference|:         {(c - c_ref[0]).abs().max().item():.2e}")

formula = 4 * hidden * (n_in + hidden) + 8 * hidden
print(f"parameters: formula 4h(x + h) + 8h = {formula}, PyTorch = {sum(p.numel() for p in ref.parameters())}")

# Gradients agree too
loss_ref = out_ref.pow(2).sum()
loss_mine = out_mine.pow(2).sum()
g_ref = torch.autograd.grad(loss_ref, ref.weight_hh_l0)[0]
g_mine = torch.autograd.grad(loss_mine, mine.W_hh)[0]
print(f"max |gradient difference| for W_hh: {(g_ref - g_mine).abs().max().item():.2e}")

# The forget gate decides how long the cell remembers: with f near 1 a value survives many steps
for f_bias in (0.0, 1.0, 3.0):
    with torch.no_grad():
        mine.b_ih.zero_(); mine.b_hh.zero_()
        mine.b_ih[hidden:2 * hidden] = f_bias      # the forget gate's slice of the bias
        mine.W_ih.mul_(0.0); mine.W_hh.mul_(0.0)   # isolate the effect of the bias
        h = torch.zeros(1, hidden); c = torch.ones(1, hidden)
        for t in range(50):
            h, c = mine(torch.zeros(1, n_in), (h, c))
    print(f"forget-gate bias {f_bias:.0f}: f = {torch.sigmoid(torch.tensor(f_bias)).item():.3f}, "
          f"cell value left after 50 steps = {c.mean().item():.2e}")
