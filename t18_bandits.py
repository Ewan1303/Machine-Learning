"""Test 18: Bandits - learning from reward when the world changes.

A k-armed bandit is the simplest setting where an agent has to learn behaviour from
reward alone: choose one of k actions, get a noisy reward, repeat. The agent keeps an
estimate Q(a) of each action's value and updates it after every reward:

  sample average   Q <- Q + (1/n)(R - Q)    every past reward counts equally
  constant step    Q <- Q + alpha(R - Q)    recent rewards count more; old ones fade

It also has to explore: epsilon-greedy picks a random arm with probability epsilon,
UCB adds a bonus to arms it has tried less often.

Two worlds:
  stationary   each arm's true value is fixed (drawn once from N(0, 1))
  drifting     all arms start equal and every true value takes a small random walk
               (standard deviation 0.01 per step), so the best arm keeps changing

Run:   python t18_bandits.py          (about 25 s)
       python t18_bandits.py --quick  (fewer simulated problems, a few seconds)

Look for: in the stationary world a little exploration beats pure greed, and sample
averages work well; in the drifting world the constant step size wins, because an
agent that never forgets keeps trusting rewards from a world that no longer exists.
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlguide_data import results_dir


def run_bandit(n_runs, n_steps, k=10, epsilon=0.1, step_size=None, ucb_c=None, drift=0.0, seed=0):
    """Simulate n_runs independent bandit problems at once (vectorised over runs).

    step_size=None means sample averages (alpha = 1/n); a number means a constant alpha.
    ucb_c=None means epsilon-greedy; a number means UCB selection with that c.
    drift > 0 makes every true value take a Gaussian random walk with that std per step.
    Returns (mean reward at each step, fraction of runs that chose the best arm at each step).
    """
    rng = np.random.default_rng(seed)
    q_true = np.zeros((n_runs, k)) if drift > 0 else rng.normal(0, 1, (n_runs, k))
    Q = np.zeros((n_runs, k))      # the agent's value estimates
    N = np.zeros((n_runs, k))      # how many times each arm has been pulled
    rows = np.arange(n_runs)
    mean_reward = np.zeros(n_steps)
    best_chosen = np.zeros(n_steps)
    for t in range(n_steps):
        if ucb_c is not None:
            bonus = ucb_c * np.sqrt(np.log(t + 1) / np.maximum(N, 1e-12))
            a = np.where(N == 0, np.inf, Q + bonus).argmax(1)     # untried arms first
        else:
            greedy = (Q + rng.uniform(0, 1e-9, Q.shape)).argmax(1)  # random tie-breaking
            explore = rng.random(n_runs) < epsilon
            a = np.where(explore, rng.integers(0, k, n_runs), greedy)
        r = q_true[rows, a] + rng.normal(0, 1, n_runs)              # noisy reward
        N[rows, a] += 1
        alpha = 1.0 / N[rows, a] if step_size is None else step_size
        Q[rows, a] += alpha * (r - Q[rows, a])                       # the learning rule
        mean_reward[t] = r.mean()
        best_chosen[t] = (a == q_true.argmax(1)).mean()
        if drift > 0:
            q_true += rng.normal(0, drift, q_true.shape)
    return mean_reward, best_chosen


def smooth(v, k):
    return np.convolve(v, np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer simulated problems")
    args = ap.parse_args()
    runs = 300 if args.quick else 1000
    t0 = time.time()

    stationary = [
        ("greedy, sample average", dict(epsilon=0.0)),
        ("eps=0.01, sample average", dict(epsilon=0.01)),
        ("eps=0.1, sample average", dict(epsilon=0.1)),
        ("UCB c=2, sample average", dict(ucb_c=2.0)),
    ]
    drifting = [
        ("eps=0.1, sample average", dict(epsilon=0.1)),
        ("eps=0.1, constant alpha=0.1", dict(epsilon=0.1, step_size=0.1)),
        ("eps=0.1, constant alpha=0.01", dict(epsilon=0.1, step_size=0.01)),
    ]
    S_STEPS, D_STEPS = 2000, 10000
    res_s = {n: run_bandit(runs, S_STEPS, seed=1, **kw) for n, kw in stationary}
    res_d = {n: run_bandit(runs, D_STEPS, drift=0.01, seed=2, **kw) for n, kw in drifting}

    print(f"{runs} independent 10-armed bandit problems per agent\n")
    print(f"Stationary world, last 500 of {S_STEPS} steps:")
    print(f"  {'agent':30s} {'mean reward':>12s} {'best arm chosen':>16s}")
    for n, (r, o) in res_s.items():
        print(f"  {n:30s} {r[-500:].mean():12.3f} {100 * o[-500:].mean():15.1f}%")
    print(f"\nDrifting world, last 2000 of {D_STEPS} steps:")
    print(f"  {'agent':30s} {'mean reward':>12s} {'best arm chosen':>16s}")
    for n, (r, o) in res_d.items():
        print(f"  {n:30s} {r[-2000:].mean():12.3f} {100 * o[-2000:].mean():15.1f}%")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    cols = ["#7F7F7F", "#8172B2", "#3B6EA8", "#4C9A5F"]
    for (n, (r, o)), c in zip(res_s.items(), cols):
        a1.plot(smooth(100 * o, 20), color=c, lw=1.5, label=n)
    a1.set_title("Stationary world", fontsize=10)
    a1.set_xlabel("step")
    a1.set_ylabel("% of runs choosing the best arm")
    a1.set_ylim(0, 100)
    a1.legend(fontsize=7.5, frameon=False, loc="lower right")
    cols = ["#3B6EA8", "#E07B39", "#C44E52"]
    for (n, (r, o)), c in zip(res_d.items(), cols):
        a2.plot(smooth(100 * o, 200), color=c, lw=1.5, label=n)
    a2.set_title("Drifting world (true values random-walk)", fontsize=10)
    a2.set_xlabel("step")
    a2.set_ylim(0, 100)
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8.5)
    fig.tight_layout()
    out = results_dir() / "t18_bandits.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out.name}. Run time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
