"""Test 19: Q-learning, from a table to a neural network.

An agent learns to cross a small gridworld (written from scratch below) to a goal,
avoiding a pit, from reward alone. Moves are slippery: 10% of the time the agent
moves in a random direction instead of the one it chose.

Part 1, tabular Q-learning: one number Q(s, a) per state and action, updated by
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Part 2, a neural Q-network that sees only the agent's (row, column) position, trained
three ways, with three seeds each and the same number of environment steps:
    online             learn from each transition once, as it happens
    replay             store transitions in a buffer and learn from random mini-batches
    replay + target    as replay, but compute targets with a copy of the network that
                       is only updated every 200 steps (the two ideas behind DQN)
Progress is measured two ways: the error between the network's values and the true
values Q* (found by value iteration, using rules the agents never see), and how often
the greedy policy reaches the goal.

Run:   python t19_q_learning.py           (about 1 minute; runs in parallel processes)
       python t19_q_learning.py --quick   (one seed of each, about 30 s)

Look for: the table learns a safe route; the online network never learns the values;
replay learns them but with sudden spikes of error; replay plus a target network
improves slowly and steadily.
"""
import argparse
import copy
import multiprocessing
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlguide_data import results_dir, set_seed

# --------------------------------------------------------------------------- the world
LAYOUT = [
    "S..#....",
    ".#.#.##.",
    ".#...#..",
    ".####.#.",
    "......#.",
    ".##.P...",
    "....#.#G",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
ARROWS = ["^", "v", "<", ">"]


class GridWorld:
    """A slippery grid. Reward +1 at G, -1 in the pit P, 0 otherwise; both end the episode."""

    def __init__(self, layout=LAYOUT, slip=0.1, max_steps=100, seed=0):
        self.grid = [list(r) for r in layout]
        self.H, self.W = len(layout), len(layout[0])
        self.slip, self.max_steps = slip, max_steps
        self.rng = np.random.default_rng(seed)
        self.start = self._find("S")
        self.n_states = self.H * self.W

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            if ch in row:
                return (r, row.index(ch))

    def reset(self):
        self.pos, self.t = self.start, 0
        return self.state()

    def state(self):
        return self.pos[0] * self.W + self.pos[1]

    def step(self, a):
        if self.rng.random() < self.slip:
            a = self.rng.integers(4)                     # slipped
        dr, dc = ACTIONS[a]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.H and 0 <= c < self.W and self.grid[r][c] != "#":
            self.pos = (r, c)                            # walls and edges block movement
        self.t += 1
        cell = self.grid[self.pos[0]][self.pos[1]]
        reward = 1.0 if cell == "G" else (-1.0 if cell == "P" else 0.0)
        terminal = cell in "GP"
        truncated = self.t >= self.max_steps             # time limit: not a real ending
        return self.state(), reward, terminal, truncated


def coords(states, W, H):
    """The neural agent's input: (row, column) scaled to [-1, 1]. Nothing else."""
    s = np.asarray(states)
    return np.stack([2 * (s // W) / (H - 1) - 1, 2 * (s % W) / (W - 1) - 1], -1).astype(np.float32)


# --------------------------------------------------------------------------- part 1: a table
def tabular_q(env, episodes=1500, alpha=0.2, gamma=0.95, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, 4))
    returns = []
    for _ in range(episodes):
        s, done, G = env.reset(), False, 0.0
        while not done:
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s] + rng.uniform(0, 1e-9, 4)))
            s2, r, term, trunc = env.step(a)
            target = r + (0.0 if term else gamma * Q[s2].max())   # no future after the end
            Q[s, a] += alpha * (target - Q[s, a])
            s, G, done = s2, G + r, term or trunc
        returns.append(G)
    return Q, np.array(returns)


def evaluate(env, policy, episodes=100):
    """Success rate of a greedy policy (slips still happen)."""
    wins = 0
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            s, r, term, trunc = env.step(policy(s))
            done = term or trunc
        wins += r > 0
    return wins / episodes


def optimal_q(env, gamma=0.95, sweeps=500):
    """The true Q* by value iteration, using the rules of the world (which the agents never see).

    Returns Q* with shape (states, 4) and a mask of the states where the agent can act.
    """
    H, W = env.H, env.W
    Q = np.zeros((env.n_states, 4))
    open_cell = lambda r, c: 0 <= r < H and 0 <= c < W and env.grid[r][c] != "#"
    def move(s, a):
        r, c = divmod(s, W)
        nr, nc = r + ACTIONS[a][0], c + ACTIONS[a][1]
        return nr * W + nc if open_cell(nr, nc) else s
    active = np.array([env.grid[s // W][s % W] in "S." for s in range(env.n_states)])
    for _ in range(sweeps):
        V = np.where(active, Q.max(1), 0.0)                 # no future after G or P
        new = np.zeros_like(Q)
        for s in np.flatnonzero(active):
            for a in range(4):
                for b in range(4):                          # the move actually made
                    p = (1 - env.slip) * (a == b) + env.slip / 4
                    s2 = move(s, b)
                    cell = env.grid[s2 // W][s2 % W]
                    r = 1.0 if cell == "G" else (-1.0 if cell == "P" else 0.0)
                    new[s, a] += p * (r + gamma * V[s2])
        Q = new
    return Q, active


# --------------------------------------------------------------------------- part 2: a network
def q_network():
    return nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))


def make_adam(params, lr):
    try:
        return torch.optim.Adam(params, lr=lr, fused=True)   # much faster on CPU where available
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.Adam(params, lr=lr)


def neural_q(mode, seed, total_steps=10000, gamma=0.95, lr=1e-3, batch=32, buffer_size=10000,
             train_every=1, target_every=200, eval_every=1000, q_star=None, active=None):
    """mode is 'online', 'replay' or 'replay+target'.

    Returns (steps, success rates, mean |Q - Q*| at each check, network). The network learns
    every `train_every` environment steps. 'online' learns from just the transitions that
    arrived since its last update (each one used once); the replay modes learn from a random
    mini-batch of everything in the buffer.
    """
    set_seed(seed)
    env, eval_env = GridWorld(seed=seed), GridWorld(seed=seed + 1000)
    errs = []
    XY = torch.from_numpy(coords(np.arange(env.n_states), env.W, env.H))   # input for every state
    net = q_network()
    target_net = copy.deepcopy(net)
    opt = make_adam(net.parameters(), lr)
    rng = np.random.default_rng(seed)
    # The replay buffer: the agent's memory of past experience, kept in a ring of arrays.
    S = np.zeros(buffer_size, np.int64); A = np.zeros(buffer_size, np.int64)
    R = np.zeros(buffer_size, np.float32); S2 = np.zeros(buffer_size, np.int64)
    T = np.zeros(buffer_size, np.float32)
    n_stored = 0

    def greedy(s):
        with torch.no_grad():
            return int(net(XY[s:s + 1]).argmax())

    xs, ys = [], []
    s = env.reset()
    for step in range(1, total_steps + 1):
        eps = max(0.05, 1.0 - step / (0.5 * total_steps))          # explore a lot at first
        a = int(rng.integers(4)) if rng.random() < eps else greedy(s)
        s2, r, term, trunc = env.step(a)
        k = n_stored % buffer_size                                  # overwrite the oldest
        S[k], A[k], R[k], S2[k], T[k] = s, a, r, s2, float(term)
        n_stored += 1
        idx = None
        if step % train_every == 0:
            if mode == "online":
                idx = (k - np.arange(train_every)) % buffer_size    # only the newest transitions
            elif n_stored >= batch:
                idx = rng.integers(0, min(n_stored, buffer_size), batch)  # random old memories
        if idx is not None:
            with torch.no_grad():
                bootstrap = (target_net if mode == "replay+target" else net)(XY[S2[idx]]).max(1).values
                target = torch.from_numpy(R[idx]) + gamma * bootstrap * torch.from_numpy(1.0 - T[idx])
            q = net(XY[S[idx]]).gather(1, torch.from_numpy(A[idx])[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(q, target)                     # Huber loss, as in DQN
            opt.zero_grad()
            loss.backward()
            opt.step()
        if mode == "replay+target" and step % target_every == 0:
            target_net.load_state_dict(net.state_dict())           # refresh the frozen copy
        s = env.reset() if (term or trunc) else s2
        if step % eval_every == 0:
            with torch.no_grad():
                q_all = net(XY).numpy()
            policy = q_all.argmax(1)                                # greedy action in every state
            xs.append(step)
            ys.append(evaluate(eval_env, lambda st: policy[st], episodes=40))
            if q_star is not None:                                  # how far from the true values
                errs.append(np.abs(q_all - q_star)[active].mean())
    return np.array(xs), np.array(ys), np.array(errs), net


def draw_policy(ax, env, Q):
    V = Q.max(1).reshape(env.H, env.W)
    show = np.ma.masked_array(V, mask=[[ch == "#" for ch in row] for row in env.grid])
    ax.imshow(show, cmap="viridis", vmin=-0.2, vmax=1.0)
    for r in range(env.H):
        for c in range(env.W):
            ch = env.grid[r][c]
            if ch == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#404040"))
            elif ch in "GP":
                ax.text(c, r, ch, ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if ch == "P" else "black")
            else:
                ax.text(c, r, ARROWS[int(Q[r * env.W + c].argmax())], ha="center", va="center",
                        fontsize=11, color="white")
    ax.set_xticks([]); ax.set_yticks([])


def run_one(mode, seed, q_star, active):
    """One training run, for the worker processes (one thread each, so runs do not compete)."""
    torch.set_num_threads(1)
    xs, ys, errs, _ = neural_q(mode, seed, q_star=q_star, active=active)
    return xs, ys, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one seed instead of three")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel processes for the 9 neural runs (default: CPU cores - 1; 1 = none)")
    args = ap.parse_args()
    seeds = [0] if args.quick else [0, 1, 2]
    t0 = time.time()

    env = GridWorld(seed=0)
    Q, returns = tabular_q(env)
    q_star, active = optimal_q(env)
    tab_success = evaluate(GridWorld(seed=99), lambda s: int(Q[s].argmax()), episodes=500)
    start = env.start[0] * env.W + env.start[1]
    print(f"Part 1: tabular Q-learning, {len(returns)} episodes")
    print(f"  mean return, first 100 episodes {returns[:100].mean():.2f}, last 100 {returns[-100:].mean():.2f}")
    print(f"  greedy policy reaches the goal in {100 * tab_success:.0f}% of 500 test episodes")
    print(f"  value of the start state: learned {Q[start].max():.3f}, true (value iteration) "
          f"{q_star[start].max():.3f}; mean |Q - Q*| over all states {np.abs(Q - q_star)[active].mean():.3f}\n")

    # Part 2: nine training runs (3 kinds x 3 seeds), several at a time in separate processes.
    modes = ["online", "replay", "replay+target"]
    jobs = [(m, sd, q_star, active) for m in modes for sd in seeds]
    workers = args.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.starmap(run_one, jobs)
    else:
        results = [run_one(*j) for j in jobs]
    xs = results[0][0]
    succ = {m: np.array([r[1] for j, r in zip(jobs, results) if j[0] == m]) for m in modes}
    err = {m: np.array([r[2] for j, r in zip(jobs, results) if j[0] == m]) for m in modes}
    col = lambda step: list(xs).index(step)
    print(f"Part 2: neural Q-network on (row, col) input, {len(seeds)} seed(s), {xs[-1]:,} steps each")
    print("  mean error |Q - Q*| (lower is better; the true value of the start state is "
          f"{q_star[start].max():.2f}):")
    print(f"  {'training':16s} {'at 2k':>6s} {'4k':>6s} {'6k':>6s} {'8k':>6s} {'10k':>6s} {'worst check':>12s}")
    for m in modes:
        e = err[m]
        print(f"  {m:16s} " + " ".join(f"{e[:, col(s)].mean():6.2f}" for s in (2000, 4000, 6000, 8000, 10000))
              + f" {e.max():12.2f}")
    print("  % of test episodes reaching G (greedy policy, slips still happen):")
    print(f"  {'training':16s} {'at 2k':>6s} {'4k':>6s} {'6k':>6s} {'8k':>6s} {'10k':>6s} {'last 3 checks':>14s}")
    for m in modes:
        c = 100 * succ[m]
        print(f"  {m:16s} " + " ".join(f"{c[:, col(s)].mean():5.0f}%" for s in (2000, 4000, 6000, 8000, 10000))
              + f" {c[:, -3:].mean():13.0f}%")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    draw_policy(a1, env, Q)
    a1.set_title("Tabular: max Q and greedy action", fontsize=9.5)
    cols = {"online": "#C44E52", "replay": "#E07B39", "replay+target": "#3B6EA8"}
    for m in modes:
        for row in err[m]:                                         # each seed, thin
            a2.plot(xs, row, color=cols[m], lw=0.8, alpha=0.35)
        a2.plot(xs, err[m].mean(0), color=cols[m], lw=2.2, label=m)
        a3.plot(xs, 100 * succ[m].mean(0), color=cols[m], lw=2.2, label=m)
    a2.set_yscale("log")
    a2.set_ylabel("mean |Q − Q*| (log scale)")
    a2.set_title("Error in the learned values", fontsize=9.5)
    a3.set_ylabel("% of test episodes reaching G")
    a3.set_ylim(-3, 103)
    a3.set_title("Success of the greedy policy (mean)", fontsize=9.5)
    for a in (a2, a3):
        a.set_xlabel("environment steps")
        a.legend(fontsize=7.8, frameon=False, loc="upper right" if a is a2 else "upper left")
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8.5)
    fig.tight_layout()
    out = results_dir() / "t19_q_learning.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out.name}. Run time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
