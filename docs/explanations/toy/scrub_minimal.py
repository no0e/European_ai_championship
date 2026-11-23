"""
Minimal SCRUB (Kurmanji et al., NeurIPS 2023) -- teaching version, CPU, about one minute.

This script:
  1. builds a small synthetic dataset (5 classes),
  2. trains an "original" model on all of D (it deliberately overfits),
  3. trains a "retrain" model (the oracle) on D_r only,
  4. runs SCRUB (max-steps on D_f, min-steps on D_r), keeping one checkpoint per epoch,
  5. runs SCRUB+R (rewinding) with a validation set from the same distribution as D_f,
  6. measures the errors (retain / forget / test) and a "basic" membership-inference attack.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- data
X, y = make_classification(n_samples=6000, n_features=40, n_informative=12, n_redundant=4,
                           n_classes=5, n_clusters_per_class=2, class_sep=0.9, flip_y=0.05,
                           random_state=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
X_train, y_train = X[:1500], y[:1500]          # small D -> the model memorises
X_test, y_test = X[1500:4500], y[1500:4500]    # never-seen data
X_val, y_val = X[4500:], y[4500:]              # used for rewinding

# "selective" forget set: 60 examples of class 0 (the rest of class 0 stays in D_r)
idx_c0 = (y_train == 0).nonzero().squeeze()
forget_idx = idx_c0[:60]
mask = torch.ones(len(X_train), dtype=torch.bool)
mask[forget_idx] = False
Xf, yf = X_train[~mask], y_train[~mask]          # D_f
Xr, yr = X_train[mask], y_train[mask]            # D_r


def make_model():
    return nn.Sequential(nn.Linear(40, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 5))


def batches(Xs, ys, bs, shuffle=True):
    idx = torch.randperm(len(Xs)) if shuffle else torch.arange(len(Xs))
    for i in range(0, len(Xs), bs):
        j = idx[i:i + bs]
        yield Xs[j], ys[j]


def train(model, Xs, ys, epochs=150, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in batches(Xs, ys, 64):
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def error(model, Xs, ys):
    return 100.0 * (model(Xs).argmax(1) != ys).float().mean().item()


# ----------------------------------------------------------------------------- SCRUB
def kl_teacher_student(logits_s, logits_t, T=4.0):
    """d(x; w_u) = KL( p_teacher || p_student ), at temperature T (like DistillKL in the official code)."""
    log_p_s = F.log_softmax(logits_s / T, dim=1)
    p_t = F.softmax(logits_t / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * T * T


def scrub(teacher, Xr, yr, Xf, alpha=1.0, gamma=1.0, epochs=10, max_epochs=4,
          lr=1e-3, bs_r=64, bs_f=8):
    teacher.eval()
    student = copy.deepcopy(teacher)               # w_u <- w_o
    opt = torch.optim.Adam(student.parameters(), lr=lr)   # the paper: Adam at small scale
    checkpoints = [copy.deepcopy(student.state_dict())]
    for epoch in range(epochs):
        if epoch < max_epochs:                     # ---- max-step: move away from the teacher on D_f
            for xb, _ in batches(Xf, torch.zeros(len(Xf)), bs_f):
                with torch.no_grad():
                    lt = teacher(xb)
                loss = -kl_teacher_student(student(xb), lt)
                opt.zero_grad(); loss.backward(); opt.step()
        for xb, yb in batches(Xr, yr, bs_r):       # ---- min-step: stay close on D_r
            with torch.no_grad():
                lt = teacher(xb)
            ls = student(xb)
            loss = alpha * kl_teacher_student(ls, lt) + gamma * F.cross_entropy(ls, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        checkpoints.append(copy.deepcopy(student.state_dict()))
    return student, checkpoints


def rewind(checkpoints, Xf, yf, X_val_same_dist, y_val_same_dist):
    """SCRUB+R: pick the checkpoint whose error on D_f is closest to the reference,
    i.e. the error at the end of SCRUB on a validation set from the same distribution."""
    m = make_model()
    m.load_state_dict(checkpoints[-1])
    ref = error(m, X_val_same_dist, y_val_same_dist)
    best, best_gap = None, float("inf")
    for k, sd in enumerate(checkpoints):
        m.load_state_dict(sd)
        gap = abs(error(m, Xf, yf) - ref)
        if gap < best_gap:
            best, best_gap = k, gap
    m.load_state_dict(checkpoints[best])
    return m, best, ref


# ----------------------------------------------------------------------------- basic MIA
@torch.no_grad()
def basic_mia(model, Xf, yf, Xt, yt):
    """Attacker = logistic regression on the loss. 'in' = forget, 'out' = test (same class).
    Ideal score ~ 50 % (the attacker does no better than chance)."""
    n = min(len(Xf), len(Xt))
    lf = F.cross_entropy(model(Xf[:n]), yf[:n], reduction="none").clamp(-400, 400).numpy()
    lt = F.cross_entropy(model(Xt[:n]), yt[:n], reduction="none").clamp(-400, 400).numpy()
    Xa = np.concatenate([lf, lt]).reshape(-1, 1)
    ya = np.concatenate([np.ones(n), np.zeros(n)])
    cv = StratifiedShuffleSplit(n_splits=5, random_state=0)
    return 100.0 * cross_val_score(LogisticRegression(), Xa, ya, cv=cv).mean()


if __name__ == "__main__":
    original = train(make_model(), X_train, y_train)
    retrain = train(make_model(), Xr, yr)

    student, ckpts = scrub(original, Xr, yr, Xf)
    Xv0, yv0 = X_val[y_val == 0], y_val[y_val == 0]       # validation set from the same distribution as D_f
    rewound, k, ref = rewind(ckpts, Xf, yf, Xv0, yv0)

    Xt0, yt0 = X_test[y_test == 0], y_test[y_test == 0]   # "out" test examples of the same class
    print(f"{'model':<10}{'retain':>9}{'forget':>9}{'test':>9}{'MIA':>9}")
    for name, m in [("Original", original), ("Retrain", retrain),
                    ("SCRUB", student), ("SCRUB+R", rewound)]:
        print(f"{name:<10}{error(m, Xr, yr):9.1f}{error(m, Xf, yf):9.1f}"
              f"{error(m, X_test, y_test):9.1f}{basic_mia(m, Xf, yf, Xt0, yt0):9.1f}")
    print(f"Rewinding: reference = {ref:.1f} % -> chosen checkpoint = epoch {k}")

    # trajectory (useful to plot the learning dynamics)
    m = make_model()
    print("epoch,retain,forget,val")
    for e, sd in enumerate(ckpts):
        m.load_state_dict(sd)
        print(f"{e},{error(m, Xr, yr):.2f},{error(m, Xf, yf):.2f},{error(m, X_val, y_val):.2f}")
