"""Test 13: a spiking neuron and a tiny spiking network.

Part 1 simulates one leaky integrate-and-fire (LIF) neuron in NumPy:
  - its membrane voltage when a step of input current is switched on and off,
    once below the firing threshold and once above it;
  - its firing rate against the input current (the f-I curve), compared with
    the exact formula for the LIF neuron.

Part 2 trains a small spiking network on the synthetic digits:
256 inputs (pixels turned into spike trains) -> 128 LIF neurons -> 10 LIF neurons.
It compares three set-ups:
  - surrogate gradient: true spikes going forward, a smooth stand-in for the
    derivative of the spike going backward, so every layer learns;
  - true derivative: the step function's real derivative is zero almost
    everywhere, so no gradient reaches the hidden layer and only the output
    weights learn (the hidden layer keeps its random starting weights);
  - an ordinary ReLU network of the same size, for comparison.

Run:   python t13_spiking.py          (about 1 to 2 minutes on a 4-core laptop CPU)
       python t13_spiking.py --quick  (3 epochs, about half the time)

Look for: no spikes until the current passes the threshold current, simulated
rates sitting on the formula curve, and the surrogate-gradient network beating
the network whose hidden layer cannot learn by a wide margin.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import count_params, results_dir, set_seed, synthetic_digits

# --------------------------------------------------------------------------- Part 1
TAU = 10.0          # membrane time constant (ms)
R = 10.0            # membrane resistance (megaohm), so 1 nA of current gives 10 mV
V_REST = -65.0      # resting potential (mV)
V_RESET = -65.0     # potential straight after a spike (mV)
V_TH = -50.0        # firing threshold (mV)
T_REF = 2.0         # refractory period (ms)
DT = 0.05           # simulation time step (ms)


def simulate_lif(current, dt=DT):
    """Euler simulation of LIF neurons.

    current: array of shape (steps,) or (steps, n) in nA, one column per neuron.
    Returns the voltage trace (steps, n) and a boolean spike array (steps, n).
    """
    current = np.asarray(current, dtype=float)
    if current.ndim == 1:
        current = current[:, None]
    steps, n = current.shape
    v = np.full(n, V_REST)
    refractory = np.zeros(n)                  # time left in the refractory period
    volts = np.empty((steps, n))
    spikes = np.zeros((steps, n), dtype=bool)
    for k in range(steps):
        active = refractory <= 1e-9
        # leak towards V_REST plus the input drive R*I, integrated over one step
        dv = dt / TAU * (-(v - V_REST) + R * current[k])
        v = np.where(active, v + dv, V_RESET)
        fired = active & (v >= V_TH)
        v = np.where(fired, V_RESET, v)       # reset after a spike
        refractory = np.where(fired, T_REF, np.maximum(refractory - dt, 0.0))
        spikes[k] = fired
        volts[k] = v
    return volts, spikes


def lif_rate_formula(current):
    """Exact firing rate (Hz) of the LIF neuron for a constant current (nA)."""
    current = np.asarray(current, dtype=float)
    drive = R * current                       # steady-state rise above rest (mV)
    gap = V_TH - V_REST                       # rise needed to fire (mV)
    rate = np.zeros_like(current)
    ok = drive > gap                          # below this the neuron never fires
    t_to_threshold = TAU * np.log((drive[ok] + V_REST - V_RESET) / (drive[ok] - gap))
    rate[ok] = 1000.0 / (T_REF + t_to_threshold)
    return rate


def part1():
    t = np.arange(0, 200, DT)
    on = (t >= 20) & (t < 180)
    currents = np.stack([np.where(on, 1.2, 0.0), np.where(on, 2.0, 0.0)], 1)
    volts, spikes = simulate_lif(currents)

    levels = np.round(np.arange(0.0, 6.01, 0.25), 2)
    long_t = np.arange(0, 1000, DT)
    _, sp = simulate_lif(np.tile(levels, (len(long_t), 1)))
    sim_rate = []
    for j in range(len(levels)):
        times = long_t[sp[:, j]]
        sim_rate.append(1000.0 / np.diff(times).mean() if len(times) > 2 else 0.0)
    sim_rate = np.array(sim_rate)
    formula = lif_rate_formula(levels)
    rheobase = (V_TH - V_REST) / R
    return t, currents, volts, spikes, levels, sim_rate, formula, rheobase


# --------------------------------------------------------------------------- Part 2
class SpikeFn(torch.autograd.Function):
    """Heaviside step going forward. Going backward, either a fast-sigmoid
    surrogate derivative 1 / (1 + slope*|x|)^2 or, if slope == 0, the step's
    true derivative, which is zero everywhere except exactly at the threshold."""

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        if ctx.slope == 0:
            return torch.zeros_like(grad_out), None
        surrogate = 1.0 / (1.0 + ctx.slope * x.abs()) ** 2
        return grad_out * surrogate, None


class SpikingNet(nn.Module):
    """Two layers of LIF neurons, unrolled over time like an RNN."""

    def __init__(self, n_in=256, n_hidden=128, n_out=10, beta=0.9, threshold=1.0, slope=25.0):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.beta, self.threshold, self.slope = beta, threshold, slope

    def forward(self, x_spikes):
        """x_spikes: (T, batch, n_in) tensor of 0s and 1s."""
        steps, batch, _ = x_spikes.shape
        u1 = torch.zeros(batch, self.fc1.out_features)
        s1 = torch.zeros_like(u1)
        u2 = torch.zeros(batch, self.fc2.out_features)
        s2 = torch.zeros_like(u2)
        mem_out, spk_out, spk_hid = [], [], []
        for t in range(steps):
            # membrane = leak * old membrane + weighted input spikes - reset for last spike
            u1 = self.beta * u1 + self.fc1(x_spikes[t]) - s1.detach() * self.threshold
            s1 = SpikeFn.apply(u1 - self.threshold, self.slope)
            u2 = self.beta * u2 + self.fc2(s1) - s2.detach() * self.threshold
            s2 = SpikeFn.apply(u2 - self.threshold, self.slope)
            mem_out.append(u2)
            spk_out.append(s2)
            spk_hid.append(s1)
        return torch.stack(mem_out), torch.stack(spk_out), torch.stack(spk_hid)


def rate_encode(x, steps, generator=None):
    """Each pixel value in [0, 1] becomes the chance of a spike at every time step."""
    probs = x.unsqueeze(0).expand(steps, *x.shape)
    return (torch.rand(probs.shape, generator=generator) < probs).float()


def snn_predict(mem_out, spk_out):
    # most output spikes wins; the summed membrane breaks ties
    return (spk_out.sum(0) + 1e-3 * mem_out.mean(0)).argmax(1)


def train_snn(Xtr, ytr, Xte, yte, slope, epochs, steps, seed=0):
    set_seed(seed)
    net = SpikingNet(slope=slope)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    w1_start = net.fc1.weight.detach().clone()
    gen = torch.Generator().manual_seed(seed + 100)
    accs = []
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            mem, spk, _ = net(rate_encode(Xtr[idx], steps))
            # cross-entropy on the output membrane at every time step
            loss = sum(F.cross_entropy(mem[t], ytr[idx]) for t in range(steps)) / steps
            opt.zero_grad()
            loss.backward()
            opt.step()
        accs.append(evaluate_snn(net, Xte, yte, steps, gen.initial_seed()))
    hidden_changed = (net.fc1.weight.detach() - w1_start).abs().max().item()
    return net, accs, hidden_changed


@torch.no_grad()
def evaluate_snn(net, X, y, steps, seed):
    gen = torch.Generator().manual_seed(seed)
    mem, spk, _ = net(rate_encode(X, steps, gen))
    return (snn_predict(mem, spk) == y).float().mean().item()


def train_ann(Xtr, ytr, Xte, yte, epochs, seed=0):
    set_seed(seed)
    net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    accs = []
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append((net(Xte).argmax(1) == yte).float().mean().item())
    return net, accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer training epochs")
    parser.add_argument("--steps", type=int, default=20, help="time steps per image")
    args = parser.parse_args()
    epochs = 3 if args.quick else 6
    steps = args.steps
    start = time.time()

    # ---------------- Part 1
    t, currents, volts, spikes, levels, sim_rate, formula, rheobase = part1()
    print("Part 1: one LIF neuron")
    print(f"  threshold current (rheobase) = (V_th - V_rest) / R = {rheobase:.2f} nA")
    print(f"  spikes in 160 ms at 1.2 nA: {spikes[:, 0].sum()},  at 2.0 nA: {spikes[:, 1].sum()}")
    print("  current (nA)   simulated rate (Hz)   formula (Hz)")
    for c in (1.0, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0):
        j = int(np.argmin(np.abs(levels - c)))
        print(f"  {levels[j]:10.2f}   {sim_rate[j]:18.1f}   {formula[j]:12.1f}")

    # ---------------- Part 2
    Xtr, ytr = synthetic_digits(n_per_class=500, seed=0)
    Xte, yte = synthetic_digits(n_per_class=100, seed=1)
    Xtr = torch.from_numpy(Xtr.reshape(len(Xtr), -1))
    Xte = torch.from_numpy(Xte.reshape(len(Xte), -1))
    ytr, yte = torch.from_numpy(ytr), torch.from_numpy(yte)

    t0 = time.time()
    snn, acc_sur, changed_sur = train_snn(Xtr, ytr, Xte, yte, slope=25.0, epochs=epochs, steps=steps)
    t_sur = time.time() - t0
    t0 = time.time()
    _, acc_zero, changed_zero = train_snn(Xtr, ytr, Xte, yte, slope=0.0, epochs=epochs, steps=steps)
    t_zero = time.time() - t0
    t0 = time.time()
    _, acc_ann = train_ann(Xtr, ytr, Xte, yte, epochs=epochs)
    t_ann = time.time() - t0
    # the ReLU network is cheap to train, so also train it to convergence for a fair comparison
    t0 = time.time()
    _, acc_ann_long = train_ann(Xtr, ytr, Xte, yte, epochs=30)
    t_ann_long = time.time() - t0

    # activity and operation counts on the test set
    with torch.no_grad():
        gen = torch.Generator().manual_seed(7)
        x_sp = rate_encode(Xte, steps, gen)
        mem, spk, hid = snn(x_sp)
    in_spikes = x_sp.sum().item() / len(Xte)                 # input spikes per image
    hid_spikes = hid.sum().item() / len(Xte)                 # hidden spikes per image
    hid_rate = hid.mean().item()                             # fraction of (neuron, step) slots with a spike
    snn_adds = in_spikes * 128 + hid_spikes * 10             # each spike adds one weight per target neuron
    ann_macs = 256 * 128 + 128 * 10                          # multiply-accumulates per image

    print("\nPart 2: 256 -> 128 -> 10 network on synthetic digits "
          f"({len(Xtr)} train, {len(Xte)} test, {steps} time steps, {epochs} epochs)")
    print(f"  parameters: {count_params(snn):,} in each network")
    print("  set-up                          test accuracy   hidden weights moved   train time")
    print(f"  spiking, surrogate gradient     {acc_sur[-1]:13.3f}   {changed_sur:20.3f}   {t_sur:8.1f} s")
    print(f"  spiking, true (zero) gradient   {acc_zero[-1]:13.3f}   {changed_zero:20.3f}   {t_zero:8.1f} s")
    print(f"  ordinary ReLU network           {acc_ann[-1]:13.3f}   {'-':>20s}   {t_ann:8.1f} s")
    print(f"  ReLU network, 30 epochs         {acc_ann_long[-1]:13.3f}   {'-':>20s}   {t_ann_long:8.1f} s")
    print(f"  hidden spikes per neuron per step: {hid_rate:.3f}")
    print(f"  per test image: {in_spikes:.0f} input spikes, {hid_spikes:.0f} hidden spikes")
    print(f"  per test image: about {snn_adds:,.0f} additions in the SNN vs {ann_macs:,} "
          "multiply-accumulates in the ReLU network")

    # ---------------- figures
    out = results_dir()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    colours = ["#7F7F7F", "#3B6EA8"]
    for j, (lab, col) in enumerate(zip(["1.2 nA (below threshold)", "2.0 nA"], colours)):
        v = volts[:, j].copy()
        ax.plot(t, v, color=col, lw=1.2, label=lab)
        for ts in t[spikes[:, j]]:
            ax.plot([ts, ts], [V_TH, 0], color=col, lw=0.9)
    ax.axhline(V_TH, ls="--", color="#C44E52", lw=0.9)
    ax.text(198, V_TH + 1.5, "threshold", color="#C44E52", ha="right", fontsize=8)
    ax.axvspan(20, 180, color="#E07B39", alpha=0.07)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane voltage (mV)")
    ax.set_ylim(-68, 5)
    ax.set_title("LIF neuron, current on from 20 to 180 ms", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax = axes[1]
    fine = np.linspace(0, 6, 400)
    ax.plot(fine, lif_rate_formula(fine), color="#3B6EA8", lw=1.5, label="formula")
    ax.plot(levels, sim_rate, "o", ms=3.5, color="#E07B39", label="simulation")
    ax.axvline(rheobase, ls=":", color="#7F7F7F", lw=1)
    ax.text(rheobase + 0.1, 185, "threshold\ncurrent", fontsize=7.5, color="#7F7F7F", va="top")
    ax.set_xlabel("input current (nA)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curve", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "t13_lif.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.2, 1.3, 0.9]})
    ax = axes[0]
    ep = np.arange(1, epochs + 1)
    ax.plot(ep, acc_sur, "-o", ms=3, color="#3B6EA8", label="SNN, surrogate")
    ax.plot(ep, acc_zero, "-o", ms=3, color="#C44E52", label="SNN, true gradient")
    ax.plot(ep, acc_ann, "-o", ms=3, color="#7F7F7F", label="ReLU network")
    ax.axhline(acc_ann_long[-1], ls="--", lw=0.9, color="#7F7F7F", label="ReLU, 30 epochs")
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    k = 0
    ax = axes[1]
    raster = hid[:, k, :].numpy().T                         # (neurons, steps)
    order = np.argsort(-raster.sum(1))[:60]
    ax.imshow(raster[order], aspect="auto", cmap="Greys", interpolation="nearest", origin="lower")
    ax.set_xlabel("time step")
    ax.set_ylabel("hidden neuron (60 most active)")
    ax.set_title(f"Hidden spikes, one test digit ({int(yte[k])})", fontsize=9)
    ax = axes[2]
    counts = spk[:, k, :].sum(0).numpy()
    ax.bar(np.arange(10), counts, color=["#3B6EA8" if c != int(yte[k]) else "#4C9A5F" for c in range(10)])
    ax.set_xticks(range(10))
    ax.set_xlabel("output neuron (digit)")
    ax.set_ylabel("spike count")
    ax.set_title("Output spikes", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "t13_snn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results/t13_lif.png and results/t13_snn.png. Total time {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
