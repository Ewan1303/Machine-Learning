"""Task 12 solution: an agent that keeps learning across separate runs of the program.

    python task_12_agent.py                       # demo: 3 runs of each agent, from a clean start
    python task_12_agent.py --agent dqn --session # one more run that continues from the saved state
    python task_12_agent.py --reset               # forget everything

The world is a 6 x 6 gridworld with walls and slippery floors (10% of moves go a random
way). The agent starts top-left and is rewarded for reaching the bottom-right corner.
Each "session" is a separate Python process: it loads whatever the agent saved last time,
plays some episodes while learning, evaluates its greedy policy, and saves itself again.

Two agents:
  tabular  Q-learning with a 36 x 4 table (Watkins and Dayan 1992)
  dqn      a small neural Q-network with a replay buffer and a target network (Mnih et al. 2015)
Everything the agent needs to carry on is saved: Q-table or network weights, the target
network, the optimiser state, the replay buffer, the episode count (which sets epsilon) and
the random-number state.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIR = Path(__file__).resolve().parent.parent / "results" / "task_12_agent_state"
SIZE, START, GOAL = 6, (0, 0), (5, 5)
WALLS = {(1, 1), (1, 2), (1, 3), (3, 2), (3, 3), (3, 4), (4, 1)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]              # up, down, left, right
N_STATES, N_ACTIONS, MAX_STEPS, SLIP = SIZE * SIZE, 4, 60, 0.1


class GridWorld:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.pos, self.t = START, 0
        return self._s()

    def _s(self):
        return self.pos[0] * SIZE + self.pos[1]

    def step(self, a):
        if self.rng.random() < SLIP:                  # slippery floor
            a = int(self.rng.integers(N_ACTIONS))
        r, c = self.pos[0] + MOVES[a][0], self.pos[1] + MOVES[a][1]
        if 0 <= r < SIZE and 0 <= c < SIZE and (r, c) not in WALLS:
            self.pos = (r, c)
        self.t += 1
        done = self.pos == GOAL
        reward = 1.0 if done else -0.01
        return self._s(), reward, done or self.t >= MAX_STEPS, done


def epsilon(episodes_done):
    return max(0.05, 0.97 ** episodes_done)            # the schedule continues across sessions


def evaluate(policy, seed, n=50):
    env = GridWorld(np.random.default_rng(seed))
    wins, steps = 0, []
    for _ in range(n):
        s, done = env.reset(), False
        while not done:
            s, _, done, success = env.step(policy(s))
        wins += success
        steps.append(env.t)
    return wins / n, float(np.mean(steps))


# ------------------------------------------------------------------ tabular Q-learning
def tabular_session(episodes, alpha=0.5, gamma=0.95):
    path = STATE_DIR / "tabular.npz"
    if path.exists():
        z = np.load(path)
        Q, done_eps = z["Q"], int(z["episodes"])
    else:
        Q, done_eps = np.zeros((N_STATES, N_ACTIONS)), 0
    rng = np.random.default_rng(1000 + done_eps)        # new randomness each session, still repeatable
    env = GridWorld(rng)
    lengths = []
    for _ in range(episodes):
        s, done = env.reset(), False
        eps = epsilon(done_eps)
        while not done:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done, success = env.step(a)
            target = r if success else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])       # the Q-learning update
            s = s2
        lengths.append(env.t)
        done_eps += 1
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, Q=Q, episodes=done_eps)
    win, steps = evaluate(lambda s: int(Q[s].argmax()), seed=done_eps)
    return dict(episodes_total=done_eps, first10=float(np.mean(lengths[:10])),
                last10=float(np.mean(lengths[-10:])), greedy_success=win, greedy_steps=steps)


# ------------------------------------------------------------------ DQN with replay
def qnet():
    return nn.Sequential(nn.Linear(N_STATES, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))


def dqn_session(episodes, gamma=0.95, batch=64, capacity=5000, sync_every=100, warmup=200):
    path = STATE_DIR / "dqn.pt"
    net, target = qnet(), qnet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf = {"s": torch.zeros(capacity, dtype=torch.long), "a": torch.zeros(capacity, dtype=torch.long),
           "r": torch.zeros(capacity), "s2": torch.zeros(capacity, dtype=torch.long), "done": torch.zeros(capacity)}
    meta = {"episodes": 0, "updates": 0, "size": 0, "next": 0}
    if path.exists():
        ck = torch.load(path, weights_only=True)
        net.load_state_dict(ck["net"]); target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"]); buf = ck["buffer"]; meta = ck["meta"]
        torch.set_rng_state(ck["torch_rng"])
    else:
        torch.manual_seed(0)
        net = qnet(); target = qnet(); target.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(2000 + meta["episodes"])
    env = GridWorld(rng)
    onehot = lambda s: F.one_hot(torch.as_tensor(s), N_STATES).float()
    lengths = []
    for _ in range(episodes):
        s, done = env.reset(), False
        eps = epsilon(meta["episodes"])
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                with torch.no_grad():
                    a = int(net(onehot(s)).argmax())
            s2, r, done, success = env.step(a)
            i = meta["next"]                                   # write into the circular buffer
            buf["s"][i], buf["a"][i], buf["r"][i], buf["s2"][i], buf["done"][i] = s, a, r, s2, float(success)
            meta["next"] = (i + 1) % capacity
            meta["size"] = min(meta["size"] + 1, capacity)
            s = s2
            if meta["size"] >= warmup:                         # learn from a random batch of old experience
                idx = torch.randint(0, meta["size"], (batch,))
                q = net(onehot(buf["s"][idx])).gather(1, buf["a"][idx, None]).squeeze(1)
                with torch.no_grad():
                    q_next = target(onehot(buf["s2"][idx])).max(1).values
                    y = buf["r"][idx] + gamma * (1 - buf["done"][idx]) * q_next
                loss = F.smooth_l1_loss(q, y)                  # Huber loss
                opt.zero_grad(); loss.backward(); opt.step()
                meta["updates"] += 1
                if meta["updates"] % sync_every == 0:
                    target.load_state_dict(net.state_dict())   # refresh the slow-moving target
        lengths.append(env.t)
        meta["episodes"] += 1
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "target": target.state_dict(), "opt": opt.state_dict(),
                "buffer": buf, "meta": meta, "torch_rng": torch.get_rng_state()}, path)
    with torch.no_grad():
        greedy = lambda s: int(net(onehot(s)).argmax())
        win, steps = evaluate(greedy, seed=meta["episodes"])
    return dict(episodes_total=meta["episodes"], first10=float(np.mean(lengths[:10])),
                last10=float(np.mean(lengths[-10:])), greedy_success=win, greedy_steps=steps,
                buffer=meta["size"])


def reset():
    for f in ("tabular.npz", "dqn.pt"):
        (STATE_DIR / f).unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["tabular", "dqn"], default="tabular")
    ap.add_argument("--session", action="store_true", help="run one session and save")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)

    if args.reset:
        reset(); print("saved agents deleted"); sys.exit(0)
    if args.session:
        n = args.episodes or (60 if args.agent == "tabular" else 50)
        out = tabular_session(n) if args.agent == "tabular" else dqn_session(n)
        print(json.dumps(out))
        sys.exit(0)

    # Demo: three separate processes per agent, starting from nothing
    reset()
    for agent in ("tabular", "dqn"):
        print(f"\n{agent} agent, three separate runs of this program:")
        print(f"{'run':>4}{'episodes so far':>17}{'mean steps, first 10':>22}{'last 10':>9}"
              f"{'greedy success':>16}{'greedy steps':>14}")
        for k in range(1, 4):
            t0 = time.time()
            res = subprocess.run([sys.executable, "-P", __file__, "--agent", agent, "--session"],
                                 capture_output=True, text=True, check=True)
            r = json.loads(res.stdout.strip().splitlines()[-1])
            print(f"{k:>4}{r['episodes_total']:>17}{r['first10']:>22.1f}{r['last10']:>9.1f}"
                  f"{r['greedy_success']:>16.2f}{r['greedy_steps']:>14.1f}   ({time.time() - t0:.0f} s)")
    print("\nA fresh agent with no saved state would repeat run 1 every time.")
    print("The shortest path takes 10 steps; slips add a few. Saved state is in", STATE_DIR)
