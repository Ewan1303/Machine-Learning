"""Task 14 solution: a skeleton for a model that keeps learning, session after session.

    python task_14_capstone.py                  # demo: 5 sessions for each method, clean start
    python task_14_capstone.py --session 3      # run one session on top of the saved state
    python task_14_capstone.py --reset

The data arrive as a stream of sessions, and the distribution changes between them:
  session 1  digits 0-3
  session 2  digits 4-6            (new classes)
  session 3  digits 7-9            (new classes)
  session 4  all digits, rotated up to 35 degrees    (new conditions)
  session 5  all digits, much noisier                (new conditions)
Every session is a fresh start of the learner from its saved file: it loads its state, learns
from the new data, tests itself on every session seen so far, saves itself (keeping the last
three versions for rollback) and appends to a JSON log.

Methods:
  finetune   just train on the new data
  der++      Dark Experience Replay++ (Buzzega et al. 2020): a reservoir buffer that stores
             old inputs with their labels and the logits the model gave them at the time;
             new training adds a logit-matching term and a label term on buffer samples.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlguide_data import synthetic_digits, set_seed, results_dir

ROOT = results_dir() / "task_14_learner"
SESSIONS = {
    1: dict(classes=range(0, 4), max_rotate=12.0, noise=0.05),
    2: dict(classes=range(4, 7), max_rotate=12.0, noise=0.05),
    3: dict(classes=range(7, 10), max_rotate=12.0, noise=0.05),
    4: dict(classes=range(10), max_rotate=35.0, noise=0.05),
    5: dict(classes=range(10), max_rotate=12.0, noise=0.25),
}
CONFIG = dict(version=1, hidden=256, lr=1e-3, epochs=4, batch=64, buffer=500, alpha=0.3, beta=0.5)


def session_data(k, split):
    spec = SESSIONS[k]
    n = 300 if split == "train" else 100
    seed = 100 * k + (0 if split == "train" else 1)
    X, y = synthetic_digits(n_per_class=n, seed=seed, max_rotate=spec["max_rotate"],
                            noise=spec["noise"], classes=spec["classes"])
    return torch.from_numpy(X.reshape(len(X), -1)), torch.from_numpy(y)


class ContinualLearner:
    def __init__(self, method, config=CONFIG):
        self.method, self.cfg = method, dict(config)
        h = self.cfg["hidden"]
        self.model = nn.Sequential(nn.Linear(256, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        cap = self.cfg["buffer"]
        self.buf = {"x": torch.zeros(cap, 256), "y": torch.zeros(cap, dtype=torch.long),
                    "logits": torch.zeros(cap, 10)}
        self.buf_size, self.seen, self.session = 0, 0, 0
        self.history = []            # one entry per session: accuracies on every session so far

    # ---------------------------------------------------------------- persistence
    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        state = {"method": self.method, "config": self.cfg, "model": self.model.state_dict(),
                 "opt": self.opt.state_dict(), "buffer": self.buf, "buf_size": self.buf_size,
                 "seen": self.seen, "session": self.session,
                 "rng": {"torch": torch.get_rng_state(), "python": random.getstate()}}
        path = folder / f"session_{self.session:03d}.pt"
        torch.save(state, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)                       # atomic write
        shutil.copyfile(path, folder / "latest.pt")
        for old in sorted(folder.glob("session_*.pt"))[:-3]:         # keep three versions for rollback
            old.unlink()
        (folder / "log.json").write_text(json.dumps(self.history, indent=1))

    @classmethod
    def load(cls, folder, method):
        path = folder / "latest.pt"
        if not path.exists():
            return cls(method)
        s = torch.load(path, weights_only=True)
        if s["config"]["version"] != CONFIG["version"]:
            raise RuntimeError("saved learner was made by a different version of this code")
        obj = cls(s["method"], s["config"])
        obj.model.load_state_dict(s["model"]); obj.opt.load_state_dict(s["opt"])
        obj.buf, obj.buf_size, obj.seen, obj.session = s["buffer"], s["buf_size"], s["seen"], s["session"]
        torch.set_rng_state(s["rng"]["torch"]); random.setstate(s["rng"]["python"])
        log = folder / "log.json"
        obj.history = json.loads(log.read_text()) if log.exists() else []
        return obj

    # ---------------------------------------------------------------- learning
    def _remember(self, x, y, logits):
        for i in range(len(x)):                                     # reservoir sampling
            self.seen += 1
            j = self.buf_size if self.buf_size < self.cfg["buffer"] else random.randrange(self.seen)
            if j < self.cfg["buffer"]:
                self.buf["x"][j], self.buf["y"][j], self.buf["logits"][j] = x[i], y[i], logits[i]
                self.buf_size = max(self.buf_size, j + 1)

    def _buffer_batch(self, n):
        idx = torch.randint(0, self.buf_size, (n,))
        return self.buf["x"][idx], self.buf["y"][idx], self.buf["logits"][idx]

    def learn(self, X, y):
        self.model.train()
        B = self.cfg["batch"]
        for _ in range(self.cfg["epochs"]):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), B):
                xb, yb = X[perm[i:i + B]], y[perm[i:i + B]]
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                if self.method == "der++" and self.buf_size > 0:
                    x1, _, z1 = self._buffer_batch(B)               # match the old logits (DER)
                    loss = loss + self.cfg["alpha"] * F.mse_loss(self.model(x1), z1)
                    x2, y2, _ = self._buffer_batch(B)               # and the old labels (the ++)
                    loss = loss + self.cfg["beta"] * F.cross_entropy(self.model(x2), y2)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                if self.method == "der++":
                    self._remember(xb, yb, logits.detach())
        self.session += 1

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            return (self.model(X).argmax(1) == y).float().mean().item()


def run_session(k, method, folder):
    learner = ContinualLearner.load(folder, method)       # nothing survives in memory between sessions
    assert learner.session == k - 1, f"expected session {learner.session + 1}, got {k}"
    learner.learn(*session_data(k, "train"))
    accs = [learner.evaluate(*session_data(j, "test")) for j in range(1, k + 1)]
    learner.history.append({"session": k, "acc": accs})
    learner.save(folder)
    return accs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int)
    ap.add_argument("--method", choices=["finetune", "der++"], default="der++")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset:
        shutil.rmtree(ROOT, ignore_errors=True); print("deleted", ROOT); sys.exit(0)
    if args.session:
        print(run_session(args.session, args.method, ROOT / args.method)); sys.exit(0)

    set_seed(0)
    for method in ("finetune", "der++"):
        folder = ROOT / method
        shutil.rmtree(folder, ignore_errors=True)
        set_seed(0)
        print(f"\n{method}: accuracy on each session's test data after each session")
        print("after   " + "".join(f"{'s' + str(j):>7}" for j in range(1, 6)) + "   mean")
        for k in range(1, 6):
            accs = run_session(k, method, folder)
            print(f"  s{k}    " + "".join(f"{a:>7.2f}" for a in accs) + " " * 7 * (5 - k) + f"{np.mean(accs):>7.2f}")
    print("\nState, version history and log.json are in", ROOT)
