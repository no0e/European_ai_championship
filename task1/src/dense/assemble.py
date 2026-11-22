"""Détecteur dense — étape 3 : règle de décision choisie en CV sur les vraies images du train, masques du test,
fichier de soumission (et envoi avec --submit).

Par (méthode, régime prédit) on choisit, sur les 4 autres plis, la meilleure règle parmi :
  seuil t sur la probabilité · décision top-k (mIoU espéré maximal) · constante du régime ;
  chacune avec ou sans nettoyage des petites composantes (grille 128).
Le détecteur n'a jamais vu les vraies images du train : l'estimation CV est honnête pour le choix des règles.
Les rectangles TrustMark déjà trouvés par rect_search (précision de relecture >= 0,90) priment sur le détecteur.

  python assemble.py --runs runs/v1 [--rects "../paulin/out/rect_tm_test_s*.json"] --out out/sub_dense_v1.pt [--submit]
La clé est lue dans la variable d'environnement WMLOC_API_KEY (jamais affichée, jamais dans git).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from common import METHODS, RELEASE, best_split, global_ll, load_gt128, load_meta, mask_kind
from t1_metric import downsample_mask, miou_per_image, stratified_folds, topk_decision

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"
T_GRID = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
C_GRID = [0, 6, 24]
ACC_THR = {"trustmark": 0.90, "stegastamp": 0.90, "pixelseal": 0.95}


def clean(mask: np.ndarray, s: int) -> np.ndarray:
    """Retire les composantes de 1 et bouche les composantes de 0 de moins de s cellules."""
    if s <= 0:
        return mask
    from scipy import ndimage
    out = mask.copy()
    for val in (True, False):
        lab, n = ndimage.label(out == val)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            small = np.flatnonzero(sizes < s)
            small = small[small > 0]
            if len(small):
                out[np.isin(lab, small)] = not val
    return out


def best_rect(m: np.ndarray, box, span: int = 20, passes: int = 3):
    """Rectangle le plus proche du masque prédit m, au sens du mIoU (B3).

    Régularisation de forme pure : la vérité des images « rect » est exactement un rectangle, donc on
    remplace la frontière dentelée du top-k par le rectangle qui lui ressemble le plus. Le mIoU obtenu
    mesure du même coup à quel point la prédiction ÉTAIT rectangulaire, et sert de garde-fou.

    Pourquoi contre le masque binaire et non contre les probabilités : avec des probabilités brutes, la
    masse diffuse du fond (p ~ 0,2 sur 15 000 cellules) compte comme du positif manqué, et l'optimum
    grossit le rectangle bien au-delà de la vérité. Mesuré sur cas jouet : −0,10 de mIoU en moyenne.

    Une image sommée donne l'aire de n'importe quel rectangle en temps constant ; une descente par
    coordonnée sur les quatre bords coûte quelques centaines d'évaluations, pas une recherche en O(n^4).
    On part de la boîte englobante, et on ne garde que ce qui améliore : la descente ne peut pas dégrader.
    """
    S = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.float64)
    S[1:, 1:] = m.astype(np.float64).cumsum(0).cumsum(1)
    tot, n = float(m.sum()), float(m.size)

    def miou(y0, y1, x0, x1):
        if y1 <= y0 or x1 <= x0:
            return -1.0
        inter = S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]
        aire = float((y1 - y0) * (x1 - x0))
        fp, fn = aire - inter, tot - inter
        tn = n - aire - fn
        return 0.5 * (inter / max(inter + fp + fn, 1e-9) + tn / max(tn + fp + fn, 1e-9))

    y0, y1, x0, x1 = box
    best = miou(y0, y1, x0, x1)
    H, W = m.shape
    for _ in range(passes):
        avant = best
        for i in range(4):
            cur = [y0, y1, x0, x1]
            lo, hi = (0, H) if i < 2 else (0, W)
            for v in range(max(lo, cur[i] - span), min(hi, cur[i] + span) + 1):
                c = list(cur); c[i] = v
                q = miou(*c)
                if q > best:
                    best, (y0, y1, x0, x1) = q, tuple(c)
        if best <= avant + 1e-12:
            break
    return (y0, y1, x0, x1), best


def apply_rule(p: np.ndarray, rule, regime_full: bool) -> np.ndarray:
    kind, t, s = rule
    if kind == "const":
        return np.full(p.shape, regime_full, bool)
    if kind == "topkc":                                # top-k sur probabilités recalées : σ(logit(p) / T + b)
        T, b = t
        q = np.clip(p, 1e-4, 1 - 1e-4)
        m = topk_decision(1 / (1 + np.exp(-(np.log(q / (1 - q)) / T + b)))).astype(bool)
    else:
        m = (p >= t) if kind in ("thr", "rthr") else topk_decision(p).astype(bool)
    m = clean(m, s)
    if kind == "orect" and m.any():                    # B3 : remplacement CONDITIONNEL par un rectangle
        from scipy import ndimage
        lab, nc = ndimage.label(m)
        big = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
        ys, xs = np.nonzero(big)
        box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        if big.sum() >= 0.8 * m.sum():
            # t = qualité minimale de l'ajustement. Le mIoU du meilleur rectangle contre le masque prédit
            # EST la mesure de rectangularité : proche de 1 pour un vrai rectangle, nettement plus bas
            # pour une ellipse ou une tache. C'est la sélection hors pli qui fixe t.
            (y0, y1, x0, x1), sc = best_rect(m, box)
            if sc >= t:
                m = np.zeros_like(m); m[y0:y1, x0:x1] = True
    if kind == "rthr" and m.any():                     # zone presque rectangulaire -> son rectangle englobant
        from scipy import ndimage
        lab, n = ndimage.label(m)
        big = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
        ys, xs = np.nonzero(big)
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        if big.sum() >= 0.8 * m.sum() and big.sum() >= 0.72 * big[box].size:
            m = np.zeros_like(m); m[box] = True
    return m


# orect (B3) : base top-k, puis rectangle optimisé si la forme prédite est assez rectangulaire. Le
# paramètre est le seuil de rectangularité (part de la boîte englobante réellement remplie), pas un seuil
# de probabilité. Peu de variantes : au-delà, c'est le bruit de sélection qui gagne (leçon de C11).
RULES = ([("const", 0, 0)] + [("thr", t, s) for t in T_GRID for s in C_GRID] + [("topk", 0, s) for s in C_GRID]
         + [("rthr", t, 24) for t in T_GRID] + [("orect", g, s) for g in (0.92, 0.96) for s in (6, 24)])
# calage des probabilités avant le top-k (piste C11) : ajouté seulement avec --calib (sinon RULES d'origine)
CALIB_RULES = [("topkc", (T, b), s) for T, b in ((0.5, 0), (2, 0), (1, -1), (1, 1), (0.5, -1), (0.5, 1), (2, -1), (2, 1))
               for s in (6, 24)]


def load_run(run_dirs, split):
    """{nom: proba 128×128}. Plusieurs dossiers contenant la même méthode : moyenne des probabilités (ensemble)."""
    out = {}
    for meth in METHODS:
        acc = {}
        for d in run_dirs:
            f = Path(d) / f"{meth}_{split}.npz"
            if f.exists():
                z = np.load(f)
                for n, p in zip(z["name"].tolist(), z["probs"]):
                    acc.setdefault(n, []).append(p.astype(np.float32))
        out.update({n: np.mean(v, 0) for n, v in acc.items()})
    return out


# ------------------------------------------------------------------ post-traitement de la carte (A, piste C12)
def img128(split, names):
    """Images réduites en 128×128 (moyenne 4×4), RGB dans [0, 1] ; mises en cache (cache/img128_<split>.npz)."""
    from common import WORK, load_img
    f = WORK / "cache" / f"img128_{split}.npz"
    cache = {}
    if f.exists():
        z = np.load(f)
        cache = dict(zip(z["name"].tolist(), z["img"]))
    miss = [n for n in names if n not in cache]
    for n in miss:
        cache[n] = (load_img(split, n).astype(np.float32).reshape(128, 4, 128, 4, 3).mean((1, 3)) / 255).astype(np.float16)
    if miss:
        f.parent.mkdir(exist_ok=True)
        np.savez_compressed(f, name=np.array(list(cache)), img=np.stack(list(cache.values())))
    return {n: cache[n].astype(np.float32) for n in names}


def guided_gray(I, p, r, eps):
    """Filtre guidé (He et al.) : lisse p en suivant les contours de l'image I (gris)."""
    from scipy.ndimage import uniform_filter
    box = lambda x: uniform_filter(x, size=2 * r + 1, mode="reflect")
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    return box(a) * I + box(mp - a * mI)


def guided_color(I, p, r, eps):
    """Filtre guidé à guide couleur (I : 128×128×3) : suit mieux les contours d'objets que le gris."""
    from scipy.ndimage import uniform_filter
    box = lambda x: uniform_filter(x, size=(2 * r + 1, 2 * r + 1) + (1,) * (x.ndim - 2), mode="reflect")
    mI, mp = box(I), box(p)
    cov = box(I * p[..., None]) - mI * mp[..., None]
    var = box(I[..., :, None] * I[..., None, :]) - mI[..., :, None] * mI[..., None, :] + eps * np.eye(3)
    a = np.linalg.solve(var, cov[..., None])[..., 0]
    b = mp - (a * mI).sum(-1)
    return (box(a) * I).sum(-1) + box(b)


def smooth_logit(p, sigma):
    """Lissage gaussien des LOGITS (B1), isotrope : ne suppose aucun contour.

    Le filtre guidé suppose que la frontière de la zone suit les contours de l'image. C'est vrai pour les
    objets (rapport de gradient mesuré 2,89, 83 % des images au-dessus de 1,5) mais FAUX pour les trous des
    images full (1,05, 24 %) : ce sont des taches lisses qui ne suivent pas le contenu. Un lissage isotrope
    est donc le bon outil pour ce poste, qui pèse la moitié du score.

    Le lissage se fait sur le logit et non sur la probabilité : près de 0 et de 1 la moyenne des
    probabilités écrase les écarts, alors que celle des logits les conserve."""
    from scipy.ndimage import gaussian_filter
    q = np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)
    z = gaussian_filter(np.log(q / (1 - q)), float(sigma), mode="nearest")
    return 1.0 / (1.0 + np.exp(-z))


def post_map(p, img, spec, extra=None):
    """spec : [b<w>/]none | s:<sigma> (lissage gaussien des logits, B1) | g:<rayon>:<eps> (guide gris)
    | c:<rayon>:<eps> (guide couleur), rayon et sigma en cellules de 4 px.
    Préfixe b<w>/ : mélange cellule par cellule (1 − w)·p + w·extra avec une autre carte (ex. têtes R5), avant filtre ;
    sans carte `extra` pour cette image, p est gardée telle quelle."""
    if "/" in spec:
        w, spec = spec.split("/", 1)
        w = float(w[1:])
        if extra is not None and w > 0:
            p = ((1 - w) * p + w * extra).astype(np.float32)
    if spec == "none":
        return p
    if spec.startswith("s:"):
        return np.clip(smooth_logit(p, spec.split(":")[1]), 0, 1).astype(np.float32)
    kind, r, eps = spec.split(":")
    q = (guided_gray(img.mean(-1), p, int(r), float(eps)) if kind == "g" else guided_color(img, p, int(r), float(eps)))
    return np.clip(q, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ régime full / non-full (A, piste C13)
def logit_fit(X, y, l2=1.0, it=30):
    """Régression logistique (Newton, L2 sur variables centrées réduites) ; renvoie x -> P(full) > 0,5."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1]); R = l2 * np.diag(np.r_[0.0, np.ones(Z.shape[1] - 1)])
    for _ in range(it):
        q = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        H = Z.T @ (Z * (q * (1 - q))[:, None]) + R + 1e-9 * np.eye(len(w))
        w -= np.linalg.solve(H, Z.T @ (q - y) + R @ w)
    return lambda Xn: (np.c_[np.ones(len(Xn)), (np.asarray(Xn) - mu) / sd] @ w) > 0


class Regime:
    """ll : seuil appris sur la log-vraisemblance globale du message (d'origine) ; lld : régression logistique sur
    (ll, moyenne de la carte dense) ; oracle : vrai régime (diagnostic en CV seulement, jamais pour le test)."""

    def __init__(self, mode, ll, mp, full):
        self.mode = mode
        if mode == "ll":
            self.thr = best_split(ll, full)
        elif mode == "lld":
            self.f = logit_fit(np.c_[ll, mp], np.asarray(full, float))

    def __call__(self, ll, mp, full=None):
        if self.mode == "ll":
            return np.asarray(ll) > self.thr
        if self.mode == "lld":
            return self.f(np.c_[ll, mp])
        return np.asarray(full, bool)


def score_table(P, names, G, specs, I, B=None):
    """mIoU de chaque (post-traitement, règle) pour chaque image, sous chaque hypothèse de régime :
    Sall[rf] de taille (n images, len(specs) × len(RULES)), colonne = indice du post-traitement × len(RULES) + règle.
    Seule la règle constante dépend du régime : les autres sont calculées une fois. B : cartes à mélanger (b<w>/)."""
    nR = len(RULES)
    Sall = {rf: np.zeros((len(names), len(specs) * nR), np.float32) for rf in (True, False)}
    for vi, v in enumerate(specs):
        Pv = {n: post_map(P[n], I[n] if I else None, v, B.get(n) if B else None) for n in names}
        for j, rule in enumerate(RULES):
            for rf in ((True, False) if rule[0] == "const" else (True,)):
                pred = np.stack([apply_rule(Pv[n], rule, rf) for n in names])
                Sall[rf][:, vi * nR + j] = miou_per_image(pred.astype(np.uint8), G.astype(np.uint8))
            if rule[0] != "const":
                Sall[False][:, vi * nR + j] = Sall[True][:, vi * nR + j]
        print(f"post-traitement {v} : scores calculés", flush=True)
    return Sall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rects", nargs="*", default=[])
    ap.add_argument("--out", default="out/sub_dense.pt")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--override", nargs="*", default=[],
                    help="npz (name, masks 128×128) qui priment sur le détecteur pour leurs images (ex. têtes R5 de Paulin)")
    ap.add_argument("--only", nargs="*", default=list(METHODS), help="méthodes pour lesquelles on utilise le détecteur")
    ap.add_argument("--save_cv", default="", help="npz des scores CV par image (pour compare_cv.py)")
    ap.add_argument("--post", nargs="*", default=["none"], help="post-traitements candidats de la carte (none, "
                    "s:<sigma> lissage gaussien des logits, g/c:<rayon>:<eps> filtre guidé), choisis en CV "
                    "avec la règle par (méthode, régime) : none | g:<r>:<eps> | c:<r>:<eps> (filtre guidé gris / couleur)")
    ap.add_argument("--regime", default="ll", choices=("ll", "lld", "oracle"),
                    help="prédiction du régime : ll (d'origine), lld (ll + moyenne de la carte), oracle (diagnostic CV)")
    ap.add_argument("--calib", action="store_true", help="ajoute les règles top-k sur probabilités recalées (C11)")
    ap.add_argument("--rects_train", nargs="*", default=[], help="json de rectangles affinés sur le TRAIN (refine_local, B9) : "
                    "appliqués en CV avec la même règle que --rects sur le test (non-full prédit, précision >= ACC_THR)")
    ap.add_argument("--blend_runs", nargs="*", default=[], help="cartes à mélanger (préfixe b<w>/ dans --post), "
                    "ex. runs/r5 (têtes R5 de Paulin : <méthode>_train.npz hors pli, <méthode>_test.npz)")
    a = ap.parse_args()
    specs = list(dict.fromkeys(a.post))
    if a.calib:
        RULES.extend(CALIB_RULES)
    assert RULES[0][0] == "const"

    meta = load_meta("train"); gt = load_gt128(); ll = global_ll("train")
    P = load_run(a.runs, "train")
    P = {n: p for n, p in P.items()}
    meta = [m for m in meta if m["image_name"] in P]
    names = [m["image_name"] for m in meta]
    meths = np.array([m["method"] for m in meta]); fulls = np.array([m["full_watermark"] for m in meta])
    kinds = np.array([mask_kind(m, gt[m["image_name"]]) for m in meta])
    folds = stratified_folds([m["method"] for m in load_meta("train")], 5, 0)
    fold_of = dict(zip([m["image_name"] for m in load_meta("train")], folds))
    folds = np.array([fold_of[n] for n in names])
    lls = np.array([ll[n] for n in names])
    mps = np.array([P[n].mean() for n in names])                  # moyenne de la carte dense (pour --regime lld)
    G = np.stack([gt[n] for n in names])
    I = img128("train", names) if any(v.split("/")[-1] != "none" for v in specs) else None
    B = load_run(a.blend_runs, "train") if a.blend_runs else None

    nR = len(RULES)
    Sall = score_table(P, names, G, specs, I, B)
    ours = np.zeros(len(names)); base = np.zeros(len(names)); regime = np.zeros(len(names), bool)
    for f in range(5):
        tr, te = folds != f, folds == f
        for meth in METHODS:
            mt = tr & (meths == meth)
            if not mt.any():                           # méthode absente des runs donnés
                continue
            rp = Regime(a.regime, lls[mt], mps[mt], fulls[mt])(lls, mps, fulls)
            for rf in (True, False):
                g_tr = mt & (rp == rf); g_te = te & (meths == meth) & (rp == rf)
                if not g_te.any():
                    continue
                k = int(np.argmax(Sall[rf][g_tr].mean(0))) if g_tr.any() else 0
                ours[g_te] = Sall[rf][g_te, k]; base[g_te] = Sall[rf][g_te, 0]; regime[g_te] = rf
    print("=== CV 5 plis (règle choisie sur les autres plis), vraies images du train")
    print(f"{'méthode':<11}{'type':<7}{'n':>4}{'base R0':>9}{'dense':>8}{'gain':>8}")
    for meth in METHODS:
        for k in ("full", "objet", "rect"):
            s = (meths == meth) & (kinds == k)
            if s.any():
                print(f"{meth:<11}{k:<7}{s.sum():>4}{base[s].mean():>9.3f}{ours[s].mean():>8.3f}{ours[s].mean() - base[s].mean():>+8.3f}")
        s = meths == meth
        print(f"{meth:<11}{'TOUT':<7}{s.sum():>4}{base[s].mean():>9.3f}{ours[s].mean():>8.3f}{ours[s].mean() - base[s].mean():>+8.3f}")
    pf = [ours[folds == f].mean() for f in range(5)]
    print(f"TOUT : base {base.mean():.4f} -> dense {ours.mean():.4f} ± {np.std(pf, ddof=1):.4f} "
          f"(par pli {np.round(pf, 3).tolist()})")
    if a.save_cv:                                       # scores CV par image, pour les comparaisons appariées
        np.savez_compressed(a.save_cv, name=np.array(names), method=meths, kind=kinds, fold=folds, score=ours,
                            base=base, regime=regime)
        print(f"scores CV par image -> {a.save_cv}")

    if a.rects_train:                                   # rectangles affinés (B9) sur le train : même règle que pour le test
        rt = {}
        for pat in a.rects_train:
            for f in glob.glob(pat):
                for r in json.loads(Path(f).read_text()):
                    rt[r["image_name"]] = r
        n_acc = 0
        for i, n in enumerate(names):
            r = rt.get(n)
            if r is not None and r["acc"] >= ACC_THR[meths[i]] and not regime[i]:
                b = np.zeros((512, 512), np.uint8); y0, y1, x0, x1 = r["box"]; b[y0:y1, x0:x1] = 1
                ours[i] = miou_per_image(downsample_mask(b)[None], G[i][None].astype(np.uint8))[0]; n_acc += 1
        print(f"rectangles du train acceptés : {n_acc} (sur {len(rt)} images traitées)")
        for meth in METHODS:
            s = meths == meth
            if s.any():
                print(f"  {meth}: " + " ".join(f"{k} {ours[s & (kinds == k)].mean():.3f}" for k in ("full", "objet", "rect")
                                          if (s & (kinds == k)).any()))
        pf = [ours[folds == f].mean() for f in range(5)]
        print(f"TOUT avec rectangles : {ours.mean():.4f} ± {np.std(pf, ddof=1):.4f} (par pli {np.round(pf, 3).tolist()})")
        if a.save_cv:
            np.savez_compressed(a.save_cv, name=np.array(names), method=meths, kind=kinds, fold=folds, score=ours,
                                base=base, regime=regime)
    print(f"régime : {a.regime}, exact sur {(regime == fulls).mean():.3f} des images (CV)")
    if a.regime == "oracle":
        sys.exit("--regime oracle : diagnostic seulement, pas de fichier de test.")

    # règles finales sur tout le train
    final, reg_final, thr_reg = {}, {}, {}
    for meth in METHODS:
        s = meths == meth
        if not s.any():
            continue
        thr_reg[meth] = best_split(lls[s], fulls[s])                  # repli si la carte manque
        reg_final[meth] = Regime(a.regime, lls[s], mps[s], fulls[s])
        rp = reg_final[meth](lls, mps)
        for rf in (True, False):
            g = s & (rp == rf)
            k = int(np.argmax(Sall[rf][g].mean(0))) if g.any() else 0
            final[(meth, rf)] = (specs[k // nR], RULES[k % nR])
    print("règles finales :", {f"{m}|{'full' if r else 'non-full'}": v for (m, r), v in final.items()})

    # test
    tmeta = load_meta("test"); tll = global_ll("test"); PT = load_run(a.runs, "test")
    TI = img128("test", [m["image_name"] for m in tmeta if m["image_name"] in PT]) if I else None
    BT = load_run(a.blend_runs, "test") if a.blend_runs else None
    rects = {}
    for pat in a.rects:
        for f in glob.glob(pat):
            for r in json.loads(Path(f).read_text()):
                rects[r["image_name"]] = r
    over = {}
    for f in a.override:
        z = np.load(f)
        over.update({n: mk.astype(bool) for n, mk in zip(z["name"].astype(str).tolist(), z["masks"])})
    masks, stats = [], dict(rect=0, dense=0, const=0, override=0)
    for m in tmeta:
        n, meth = m["image_name"], m["method"]
        if n in PT and meth in reg_final:
            rf = bool(reg_final[meth](np.array([tll[n]]), np.array([PT[n].mean()]))[0])
        else:
            rf = bool(tll[n] > thr_reg.get(meth, 0))
        r = rects.get(n)
        if r is not None and r["acc"] >= ACC_THR[meth] and not rf:
            b = np.zeros((512, 512), np.uint8); y0, y1, x0, x1 = r["box"]; b[y0:y1, x0:x1] = 1
            masks.append(downsample_mask(b).astype(bool)); stats["rect"] += 1
        elif n in over:
            masks.append(over[n]); stats["override"] += 1
        elif n in PT and (meth, rf) in final and meth in a.only:
            spec, rule = final[(meth, rf)]
            masks.append(apply_rule(post_map(PT[n], TI[n] if TI else None, spec, BT.get(n) if BT else None), rule, rf))
            stats["dense"] += 1
        else:
            masks.append(np.full((128, 128), rf, bool)); stats["const"] += 1
    masks = torch.from_numpy(np.stack(masks)).bool()
    tnames = [m["image_name"] for m in tmeta]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": tnames, "masks": masks}, str(out))
    sub = torch.load(str(out), weights_only=True)
    assert len(set(sub["image_names"])) == 1020 and tuple(sub["masks"].shape) == (1020, 128, 128)
    assert sub["masks"].dtype == torch.bool and os.path.getsize(out) / 2 ** 20 <= 20
    frac = masks.float().mean((1, 2))
    print(f"OK {out} ({os.path.getsize(out) / 2 ** 20:.1f} Mo) : {stats} ; fraction prédite médiane {frac.median():.2f}, "
          f"tout-1 {int((frac == 1).sum())}, tout-0 {int((frac == 0).sum())}")
    if a.submit:
        import requests
        key = os.environ.get("WMLOC_API_KEY")
        if not key:
            sys.exit("WMLOC_API_KEY absente : rien n'est envoyé.")
        with open(out, "rb") as f:
            r = requests.post(f"{BASE_URL}/submit/{TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (out.name, f, "application/octet-stream")}, timeout=(10, 300))
        try:
            print("HTTP", r.status_code, r.json())
        except Exception:
            print("HTTP", r.status_code, r.text[:300])


if __name__ == "__main__":
    main()
