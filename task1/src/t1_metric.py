"""Tâche 1 : métrique officielle (slides 30-31), règle de décision, CV et baselines.

Tout travaille sur des tableaux numpy de grilles 128x128 (0/1), indépendamment du format des
fichiers fournis (le chargeur viendra quand on aura les données).

Point NON confirmé : le cas 0/0 (aucune cellule de la classe c, ni prédite ni vraie). Les slides
ne le traitent pas. `empty` fixe la valeur de l'IoU dans ce cas (1.0 = convention la plus courante).
À remplacer par la convention du starter kit dès qu'on l'a lue.
"""
from __future__ import annotations

import numpy as np

GRID = 128
CELL = 4


# ---------------------------------------------------------------- métrique
def iou_per_class(pred: np.ndarray, gt: np.ndarray, empty: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """pred, gt : (N, 128, 128) ou (128, 128), valeurs 0/1. Renvoie (IoU0, IoU1) par image."""
    p = np.asarray(pred, dtype=bool).reshape(-1, GRID * GRID)
    g = np.asarray(gt, dtype=bool).reshape(-1, GRID * GRID)
    out = []
    for pc, gc in ((~p, ~g), (p, g)):
        inter = (pc & gc).sum(1).astype(float)
        union = (pc | gc).sum(1).astype(float)
        out.append(np.where(union > 0, inter / np.maximum(union, 1), empty))
    return out[0], out[1]


def miou_per_image(pred, gt, empty: float = 1.0) -> np.ndarray:
    iou0, iou1 = iou_per_class(pred, gt, empty)
    return 0.5 * (iou0 + iou1)


def miou_score(pred, gt, empty: float = 1.0) -> float:
    """Score officiel : moyenne sur les images du mIoU par image."""
    return float(miou_per_image(pred, gt, empty).mean())


def downsample_mask(mask512: np.ndarray) -> np.ndarray:
    """Règle de l'énoncé : cellule 4x4 = 1 si au moins 50 % de ses pixels sont marqués.
    mask512 : (..., 512, 512), 0/1 ou fraction dans [0, 1]."""
    m = np.asarray(mask512, dtype=float)
    lead = m.shape[:-2]
    frac = m.reshape(*lead, GRID, CELL, GRID, CELL).mean(axis=(-3, -1))
    return (frac >= 0.5).astype(np.uint8)


# ------------------------------------------------- décision optimale par image
def topk_decision(prob: np.ndarray, empty: float = 1.0) -> np.ndarray:
    """Transforme une carte de probabilités (128, 128) en masque 0/1 qui maximise le mIoU ESPÉRÉ.

    Si p_j = P(cellule j marquée) et que l'on prédit 1 sur un ensemble S, alors (espérances exactes)
      E[I1] = sum_{S} p,        E[U1] = |S| + sum_{non S} p
      E[I0] = sum_{non S}(1-p), E[U0] = N - sum_{S} p
    On approche E[IoU] par E[I]/E[U] (N = 16 384 cellules : approximation raisonnable). À |S| fixé,
    le meilleur S est formé des |S| cellules de plus forte p ; on balaie donc k = 0..N.
    Avantage sur un seuil fixe : le seuil s'adapte à chaque image (petite zone -> seuil différent).
    Suppose des p calibrées (à vérifier en CV ; sinon recalibrer, p.ex. puissance ou Platt).
    """
    p = np.asarray(prob, dtype=float).ravel()
    n = p.size
    order = np.argsort(-p)
    cum = np.concatenate([[0.0], np.cumsum(p[order])])  # cum[k] = somme des k plus grandes
    k = np.arange(n + 1, dtype=float)
    tot = cum[-1]
    i1, u1 = cum, k + (tot - cum)
    i0, u0 = (n - k) - (tot - cum), n - cum
    with np.errstate(invalid="ignore", divide="ignore"):
        e1 = np.where(u1 > 0, i1 / u1, empty)
        e0 = np.where(u0 > 0, i0 / u0, empty)
    best = int(np.argmax(e0 + e1))
    out = np.zeros(n, dtype=np.uint8)
    out[order[:best]] = 1
    return out.reshape(np.shape(prob))


def threshold_decision(prob: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(prob) >= thr).astype(np.uint8)


# ------------------------------------------------------------------ CV
def stratified_folds(methods, k: int = 5, seed: int = 0) -> np.ndarray:
    """Numéro de pli (0..k-1) pour chaque image, stratifié par méthode."""
    methods = np.asarray(methods)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(methods), dtype=int)
    for m in np.unique(methods):
        idx = np.flatnonzero(methods == m)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cv_report(scores_per_image: np.ndarray, folds: np.ndarray, methods=None) -> dict:
    """Moyenne ± écart-type sur les plis (et par méthode si fourni) d'un score par image."""
    s = np.asarray(scores_per_image, dtype=float)
    per_fold = np.array([s[folds == f].mean() for f in np.unique(folds)])
    rep = {"mean": float(per_fold.mean()), "std": float(per_fold.std(ddof=1)), "folds": per_fold.round(4).tolist()}
    if methods is not None:
        methods = np.asarray(methods)
        rep["by_method"] = {str(m): float(s[methods == m].mean()) for m in np.unique(methods)}
    return rep


# ------------------------------------------------------------ baselines
def baseline_constant(n: int, value: int) -> np.ndarray:
    return np.full((n, GRID, GRID), value, dtype=np.uint8)


def baseline_mean_map(gt_train, methods_train, methods_test, decision: str = "thr",
                      empty: float = 1.0, thr_grid=np.linspace(0.05, 0.95, 19)):
    """Carte moyenne des masques par méthode, puis seuil choisi sur le train (ou décision top-k).
    Renvoie (prédictions test, seuil retenu par méthode)."""
    gt_train = np.asarray(gt_train)
    methods_train, methods_test = np.asarray(methods_train), np.asarray(methods_test)
    preds = np.zeros((len(methods_test), GRID, GRID), dtype=np.uint8)
    chosen = {}
    for m in np.unique(methods_test):
        mean_map = gt_train[methods_train == m].mean(0)
        if decision == "topk":
            chosen[str(m)] = "topk"
            preds[methods_test == m] = topk_decision(mean_map, empty)
            continue
        g = gt_train[methods_train == m]
        best_thr = max(thr_grid, key=lambda t: miou_score(
            np.broadcast_to(threshold_decision(mean_map, t), g.shape), g, empty))
        chosen[str(m)] = float(best_thr)
        preds[methods_test == m] = threshold_decision(mean_map, best_thr)
    return preds, chosen
