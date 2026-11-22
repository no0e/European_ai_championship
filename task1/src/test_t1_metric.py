"""Vérifications jouets de t1_metric (python tasks/Hackathon_Watermark_Localisation_Task/test_t1_metric.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from t1_metric import (GRID, baseline_constant, baseline_mean_map, cv_report, downsample_mask,  # noqa: E402
                       miou_per_image, miou_score, stratified_folds, topk_decision)


def rect_mask(rng, size=GRID):
    m = np.zeros((size, size), np.uint8)
    h, w = rng.integers(size // 8, size // 2, 2)
    y, x = rng.integers(0, size - h), rng.integers(0, size - w)
    m[y:y + h, x:x + w] = 1
    return m


def main():
    rng = np.random.default_rng(0)
    gt = np.stack([rect_mask(rng) for _ in range(30)])
    f = gt.reshape(30, -1).mean(1)

    # 1. prédiction parfaite -> 1
    assert np.allclose(miou_per_image(gt, gt), 1.0)
    # 2. tout à 0 -> (1-f)/2 ; tout à 1 -> f/2 (formules analytiques)
    assert np.allclose(miou_per_image(baseline_constant(30, 0), gt), (1 - f) / 2)
    assert np.allclose(miou_per_image(baseline_constant(30, 1), gt), f / 2)
    # 3. cas 0/0 : vrai masque vide et prédiction vide -> IoU1 = empty
    z = np.zeros((1, GRID, GRID), np.uint8)
    assert miou_score(z, z, empty=1.0) == 1.0 and miou_score(z, z, empty=0.0) == 0.5
    # 4. règle des 50 % : une cellule à 8/16 pixels marqués -> 1, à 7/16 -> 0
    m512 = np.zeros((512, 512), np.uint8)
    m512[0:2, 0:4] = 1          # cellule (0,0) : 8/16
    m512[0:1, 4:8] = 1          # cellule (0,1) : 4/16
    m512[1, 4:7] = 1            # + 3 -> 7/16
    d = downsample_mask(m512)
    assert d[0, 0] == 1 and d[0, 1] == 0 and d.sum() == 1
    # 5. top-k : probabilités parfaites -> masque exact.
    assert (topk_decision(gt[0].astype(float)) == gt[0]).all()
    #    Probabilités CALIBRÉES (gt tiré selon p) : top-k au moins aussi bon que le meilleur seuil fixe.
    #    (Constaté : sur des p NON calibrées, top-k perd nettement -> calibrer avant de l'utiliser.)
    P, G = [], []
    for _ in range(200):
        r = rect_mask(rng).astype(float)
        p = np.where(r > 0, rng.uniform(0.55, 0.95), rng.uniform(0.02, 0.3))
        P.append(p)
        G.append((rng.random(p.shape) < p).astype(np.uint8))
    P, G = np.stack(P), np.stack(G)
    s_topk = miou_score(np.stack([topk_decision(p) for p in P]), G)
    s_thr = max(miou_score((P >= t).astype(np.uint8), G) for t in np.linspace(0.05, 0.95, 19))
    assert s_topk >= s_thr - 0.005, (s_topk, s_thr)
    # 6. plis stratifiés : chaque méthode répartie équitablement
    methods = np.repeat(["pixelseal", "trustmark", "stegastamp"], 10)
    folds = stratified_folds(methods, k=5)
    for meth in np.unique(methods):
        assert np.bincount(folds[methods == meth], minlength=5).tolist() == [2] * 5
    rep = cv_report(miou_per_image(baseline_constant(30, 0), gt), folds, methods)
    preds, thr = baseline_mean_map(gt, methods, methods)
    print(f"OK | top-k {s_topk:.3f} vs meilleur seuil fixe {s_thr:.3f} (toy calibré) | tout-0 {rep['mean']:.3f}±{rep['std']:.3f} "
          f"| carte moyenne {miou_score(preds, gt):.3f} (seuils {thr})")


if __name__ == "__main__":
    main()
