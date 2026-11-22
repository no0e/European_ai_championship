"""Détecteur dense — étape 2 : entraînement sur composites synthétiques, validation sur les VRAIES images du
train (jamais vues à l'entraînement : seules leurs formes de masques servent), prédictions train + test.

  python train_dense.py --method pixelseal --run v1 [--steps 6000] [--bs 16]
Test rapide CPU : --steps 4 --bs 2 --workers 0 --eval_n 6 --device cpu --no_test
Sorties : runs/<run>/<méthode>.pt, <méthode>_train.npz, <méthode>_test.npz (probabilités 128×128), <méthode>_log.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import (HERE, METHODS, MaskSampler, build_model, composite, default_arch, ds128, load_gt128, load_img,
                    load_meta, mask_kind)
from t1_metric import miou_per_image, stratified_folds, topk_decision


ONES = np.ones((128, 128), np.float32)

# ------------------------------------------------------------------ k5 : mise au cadre des vraies images
# Mesuré sur le réel (frames.py) : l'image publiée est un recadrage d'un cadre tatoué plus grand (facteur s).
# Les composites synthétiques, eux, tatouent le cadre 512 entier. On remet donc chaque vraie image dans SON
# cadre (complétée par réflexion, puis ramenée en 512) : le motif retrouve l'alignement vu à l'entraînement.
FRAMES = {}                                            # nom -> (côté du cadre en px, ox, oy)


def load_frames(path, zmin=25.0):
    for n, v in json.loads(Path(path).read_text()).items():
        b = v["best"]
        if b["Z"] > zmin and b["s"] > 1.012:
            FRAMES[n] = (int(round(512 * b["s"])), int(b["ox"]), int(b["oy"]))
    print(f"cadres : {len(FRAMES)} images mises au cadre ({path})", flush=True)


def canon_img(y8, fr):
    Fs, ox, oy = fr
    can = np.pad(y8, ((oy, Fs - 512 - oy), (ox, Fs - 512 - ox), (0, 0)), mode="reflect")
    return np.asarray(Image.fromarray(can).resize((512, 512), Image.Resampling.BILINEAR))


def canon_map(a128, fr, fill=0.0):
    """Carte 128×128 de l'image -> carte 128×128 du cadre (hors image : `fill`)."""
    Fs, ox, oy = fr
    a = torch.from_numpy(np.asarray(a128, np.float32))[None, None]
    a = F.interpolate(a, size=(512, 512), mode="nearest")
    a = F.pad(a, (ox, Fs - 512 - ox, oy, Fs - 512 - oy), value=fill)
    return F.interpolate(a, size=(128, 128), mode="area")[0, 0].numpy()


def uncanon_map(p128, fr):
    """Carte 128×128 du cadre -> carte 128×128 de l'image."""
    Fs, ox, oy = fr
    a = torch.from_numpy(np.asarray(p128, np.float32))[None, None]
    a = F.interpolate(a, size=(Fs, Fs), mode="bilinear", align_corners=False)[:, :, oy:oy + 512, ox:ox + 512]
    return F.avg_pool2d(a, 4)[0, 0].numpy()


def real_item(split, n, lab128, wgt128):
    """Vraie image (train ou test pseudo-étiquetée), mise au cadre si son cadre est connu."""
    y = load_img(split, n)
    if n not in FRAMES:
        return y, lab128, wgt128
    fr = FRAMES[n]
    lab = (canon_map(lab128, fr) >= 0.5).astype(np.float32)
    wgt = canon_map(wgt128, fr)                          # hors image : poids 0 (contenu miroir, étiquette inconnue)
    return canon_img(y, fr), lab, (wgt >= 0.999).astype(np.float32)


def pseudo_labels(files, band):
    """Pseudo-étiquettes du test, à partir de fichiers de modèles qui n'ont PAS vu le pli validé :
    - fichiers de probabilités (clé « probs ») : moyenne, puis décision top-k ;
    - fichiers de masques (clé « masks », make_pseudo.py : post-traitement + règle choisis hors pli) : tels quels.
    Les cellules à moins de `band` cellules de la frontière prédite sont ignorées (poids 0).
    Retourne [(nom, étiquette 128 float32, poids 128 float32)]."""
    from scipy import ndimage
    acc, given = {}, {}
    for f in files:
        z = np.load(f)
        if "masks" in z.files:
            given.update({n: m.astype(bool) for n, m in zip(z["name"].tolist(), z["masks"])})
            continue
        for n, p in zip(z["name"].tolist(), z["probs"]):
            acc.setdefault(n, []).append(p.astype(np.float32))
    out = []
    for n in list(given) + [n for n in acc if n not in given]:
        lab = given[n] if n in given else topk_decision(np.mean(acc[n], 0)).astype(bool)
        if lab.all() or not lab.any():
            wgt = ONES.copy()
        else:
            d_in, d_out = ndimage.distance_transform_edt(lab), ndimage.distance_transform_edt(~lab)
            wgt = np.where(lab, d_in > band, d_out > band).astype(np.float32)
        out.append((n, lab.astype(np.float32), wgt))
    return out


def degrade(w, rng):
    """A1c : version dégradée de l'image TATOUÉE (flou, JPEG, ou réduction puis agrandissement).

    Sert à remplir les trous quand on fait l'hypothèse qu'ils gardent un reste de signal. Cette hypothèse
    n'est PAS tranchée (le contenu des trous est continu avec l'image, cf. TODO_0.80_PLUS.md § 2) : ne
    l'activer qu'en mélange, et seulement si le diagnostic A0 va dans ce sens."""
    u = rng.random()
    if u < 0.34:
        from scipy.ndimage import gaussian_filter
        s = float(rng.uniform(1.0, 3.0))
        return np.clip(gaussian_filter(w.astype(np.float32), (s, s, 0)), 0, 255).astype(np.uint8)
    if u < 0.67:
        import io
        b = io.BytesIO()
        Image.fromarray(w).save(b, "JPEG", quality=int(rng.integers(50, 91)))
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"))
    f = float(rng.uniform(0.3, 0.7))
    im = Image.fromarray(w)
    k = max(8, int(round(512 * f)))
    return np.asarray(im.resize((k, k), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.BILINEAR))


class SynthSet(Dataset):
    """Chaque exemple : (image uint8 3×512×512, cible 128×128, poids 128×128 ; poids 0 = cellule ignorée).

    Tirage : vraie image du train (real_frac), image du test pseudo-étiquetée (pseudo_frac), sinon composite
    synthétique. Composites : la zone NON tatouée vient de l'image elle-même (x), ou avec p_fill d'une AUTRE image
    de base (objet ajouté par-dessus / objet transplanté dans une autre image, comme dans l'énoncé), ou avec p_blur
    (images de type full) d'une version lissée de x (trou « inpainté »)."""

    def __init__(self, method, pairs_dir, n, seed=0, p_rect=0.0, real=None, real_frac=0.0, pseudo=None,
                 pseudo_frac=0.0, p_fill=0.0, p_blur=0.0, p_wdeg=0.0, alpha=(1.0, 1.0)):
        self.dir = Path(pairs_dir) / method
        # les images tenues à l'écart (validation synthétique fixe, A2) ne servent jamais à l'entraînement
        self.index = [it for it in json.loads((self.dir / "index.json").read_text()) if not it.get("holdout")]
        if not self.index:
            raise SystemExit(f"aucune paire d'entraînement dans {self.dir} (toutes marquées holdout ?)")
        self.p_wdeg, self.alpha = p_wdeg, tuple(alpha)
        self.n, self.seed, self.sampler = n, seed, None
        self.real, self.real_frac = real or [], real_frac if real else 0.0   # [(nom, masque 128)] des plis d'entraînement
        self.pseudo, self.pseudo_frac = pseudo or [], pseudo_frac if pseudo else 0.0
        self.p_fill, self.p_blur = p_fill, p_blur
        self.p_rect = p_rect if any(it.get("rects") for it in self.index) else 0.0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.sampler is None:                                   # un tirage indépendant par worker
            self.sampler = MaskSampler()
            wi = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, wi.id if wi else 0, i])
            self.X = np.load(self.dir / "x.npy", mmap_mode="r"); self.W = np.load(self.dir / "w.npy", mmap_mode="r")
            self.R = np.load(self.dir / "r.npy", mmap_mode="r") if (self.dir / "r.npy").exists() else None
        t = lambda a: torch.from_numpy(np.array(a))               # copie : les tableaux lus sont en lecture seule
        u = self.rng.random()
        if u < self.real_frac:                                     # vraie image du train (aucune augmentation :
            n, g = self.real[int(self.rng.integers(len(self.real)))]         # un retournement changerait le tatouage)
            yy, lab, wgt = real_item("train", n, g.astype(np.float32), ONES)
            return t(yy).permute(2, 0, 1), t(lab), t(wgt)
        if u < self.real_frac + self.pseudo_frac:                  # image du test, pseudo-étiquetée
            n, lab, wgt = self.pseudo[int(self.rng.integers(len(self.pseudo)))]
            yy, lab, wgt = real_item("test", n, lab, wgt)
            return t(yy).permute(2, 0, 1), t(lab), t(wgt)
        it = self.index[int(self.rng.integers(len(self.index)))]
        x = np.asarray(self.X[it["idx"]])
        if self.p_rect and self.R is not None and it.get("rects") and self.rng.random() < self.p_rect:
            y0, y1, x0, x1, off = it["rects"][int(self.rng.integers(len(it["rects"])))]   # type rectangle (outpainting)
            h, w = y1 - y0, x1 - x0
            y = x.copy(); y[y0:y1, x0:x1] = np.asarray(self.R[off:off + h * w * 3]).reshape(h, w, 3)
            M = np.zeros((512, 512), bool); M[y0:y1, x0:x1] = True
            return t(y).permute(2, 0, 1), t(ds128(M).astype(np.float32)), t(ONES)
        w = np.asarray(self.W[it["idx"]])
        M, kind = self.sampler.sample(self.rng, it["name"] if it["split"] != "div2k" else None, it["flip"])
        only = os.environ.get("K5_KIND")                           # k5 : modèle spécialisé (objet = non-full seulement, full = trous seulement)
        k_ = 0
        while only and kind != only and k_ < 20:
            M, kind = self.sampler.sample(self.rng, it["name"] if it["split"] != "div2k" else None, it["flip"]); k_ += 1
        fill, v = x, self.rng.random()
        if v < self.p_fill:                                        # zone non tatouée = contenu d'une autre image
            other = self.index[int(self.rng.integers(len(self.index)))]
            if other["name"] != it["name"]:
                fill = np.asarray(self.X[other["idx"]])
        elif v < self.p_fill + self.p_blur and kind == "full":    # trou lissé (inpainting sans détail)
            from scipy.ndimage import gaussian_filter
            s = float(self.rng.uniform(2, 6))
            fill = np.clip(gaussian_filter(x.astype(np.float32), (s, s, 0)), 0, 255).astype(np.uint8)
        elif v < self.p_fill + self.p_blur + self.p_wdeg and kind == "full":
            fill = degrade(w, self.rng)                            # A1c : trou = image TATOUÉE dégradée
        a = 1.0                                                    # A1b : une fois sur deux, force changée
        if self.alpha != (1.0, 1.0) and self.rng.random() < 0.5:
            a = float(self.rng.uniform(*self.alpha))
        y = composite(fill, w, M, self.rng, a)
        return t(y).permute(2, 0, 1), t(ds128(M).astype(np.float32)), t(ONES)


class SynthVal(Dataset):
    """A2 — validation synthétique fixe : composites à graine fixe sur les images de base tenues à l'écart.

    La validation réelle mélange deux causes de plafond (capacité du modèle, écart synthétique / réel) ; celle-ci
    n'en garde qu'une. Si elle monte alors que la réelle stagne, c'est le domaine ; si elle stagne aussi, c'est
    le signal ou la capacité, et allonger l'entraînement ne sert plus à rien."""

    def __init__(self, method, pairs_dir, n=48, seed=777):
        d = Path(pairs_dir) / method
        idx = [it for it in json.loads((d / "index.json").read_text()) if it.get("holdout")]
        self.items = []
        if not idx:
            return
        rng = np.random.default_rng(seed)
        X = np.load(d / "x.npy", mmap_mode="r"); W = np.load(d / "w.npy", mmap_mode="r")
        sampler = MaskSampler()
        for i in range(n):
            it = idx[i % len(idx)]
            x = np.asarray(X[it["idx"]]); w = np.asarray(W[it["idx"]])
            kind = "full" if i % 2 == 0 else "objet"
            while True:
                M, k = sampler.sample(rng)
                if k == kind:
                    break
            self.items.append((composite(x, w, M, rng), ds128(M), kind))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.from_numpy(self.items[i][0].copy()).permute(2, 0, 1), i


def evaluate_synth(probs, sv):
    rows = {}
    for i, (_, g, kind) in enumerate(sv.items):
        p = probs[i].astype(np.float32)
        rows.setdefault(kind, []).append((auc(p.ravel(), g.ravel()),
                                          float(miou_per_image(topk_decision(p)[None], g[None].astype(np.uint8))[0])))
    rep, tot = {}, []
    for k, r in rows.items():
        r = np.array(r); tot.append(r)
        rep[k] = dict(n=len(r), auc_med=float(np.nanmedian(r[:, 0])), miou_topk=float(r[:, 1].mean()))
    t = np.concatenate(tot)
    rep["tout"] = dict(n=len(t), auc_med=float(np.nanmedian(t[:, 0])), miou_topk=float(t[:, 1].mean()))
    return rep


class RealSet(Dataset):
    def __init__(self, split, method, limit=0, names=None):
        self.split = split
        self.meta = [m for m in load_meta(split) if m["method"] == method and (names is None or m["image_name"] in names)]
        if limit:
            self.meta = self.meta[::max(1, len(self.meta) // limit)][:limit]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        n = self.meta[i]["image_name"]; y = load_img(self.split, n)
        if n in FRAMES:
            y = canon_img(y, FRAMES[n])
        return torch.from_numpy(y.copy()).permute(2, 0, 1), i


def retry(fn, what, n=40, wait=20):
    """Le quota du projet partagé sature par moments (« Disk quota exceeded ») : on réessaie."""
    for _ in range(n):
        try:
            return fn()
        except (OSError, RuntimeError) as e:
            print(f"écriture refusée ({what}) : {e} ; nouvel essai dans {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"écriture impossible : {what}")


def loss_fn(logits, y, wgt=None, miou_w=0.5):
    """BCE + miou_w · (1 − mIoU souple à deux classes, par image) : proche de la métrique officielle.
    wgt : poids par cellule (0 = ignorée, pour les pseudo-étiquettes) ; None = tout à 1 (perte d'origine).
    miou_w (A4b) : 0,5 = perte d'origine ; 1,0 pèse davantage la classe « trou », minoritaire en cellules."""
    if wgt is None:
        wgt = torch.ones_like(y)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * wgt).sum() / wgt.sum().clamp(min=1.0)
    p = torch.sigmoid(logits).flatten(1); t = y.flatten(1); w = wgt.flatten(1)
    i1 = (p * t * w).sum(1); u1 = ((p + t - p * t) * w).sum(1)
    i0 = ((1 - p) * (1 - t) * w).sum(1); u0 = (((1 - p) + (1 - t) - (1 - p) * (1 - t)) * w).sum(1)
    miou = 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1))
    return bce + miou_w * (1 - miou).mean()


@torch.no_grad()
def predict(model, ds, device, bs=32, workers=4):
    model.eval()
    out = np.zeros((len(ds), 128, 128), np.float16)
    for x, idx in DataLoader(ds, bs, shuffle=False, num_workers=workers):
        xf = x.to(device).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(xf).float()
            views = os.environ.get("K5_TTA", "")                 # k5 : vues supplémentaires « cadre plus grand » (s:fx:fy,...)
            if views:
                acc = [lg]
                for v in views.split(","):
                    s_, fx, fy = (float(t_) for t_ in v.split(":"))
                    Fs = int(round(512 * s_)); r = Fs - 512; ox, oy = int(round(r * fx)), int(round(r * fy))
                    can = F.interpolate(F.pad(xf, (ox, r - ox, oy, r - oy), mode="reflect"), size=(512, 512), mode="bilinear",
                                        align_corners=False, antialias=True)
                    l2 = model(can).float()[:, None]
                    l2 = F.interpolate(l2, size=(Fs, Fs), mode="bilinear", align_corners=False)[:, :, oy:oy + 512, ox:ox + 512]
                    acc.append(F.avg_pool2d(l2, 4)[:, 0])
                lg = torch.stack(acc).mean(0)
            p = torch.sigmoid(lg)
        out[idx.numpy()] = p.cpu().numpy().astype(np.float16)
    if FRAMES and hasattr(ds, "meta"):                    # retour dans le repère de l'image
        for i, m in enumerate(ds.meta):
            if m["image_name"] in FRAMES:
                out[i] = uncanon_map(out[i].astype(np.float32), FRAMES[m["image_name"]]).astype(np.float16)
    return out


def auc(score, y):
    from scipy.stats import rankdata
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(probs, ds, gt):
    """Par type d'image : AUC pixel (par image, médiane), mIoU au seuil 0,5 et avec la décision top-k."""
    rows = {}
    for i, m in enumerate(ds.meta):
        g = gt[m["image_name"]]
        kind = mask_kind(m, g)
        p = probs[i].astype(np.float32)
        base = np.ones_like(g) if kind == "full" else np.zeros_like(g)
        rows.setdefault(kind, []).append((
            auc(p.ravel(), g.ravel()),
            float(miou_per_image((p >= 0.5)[None].astype(np.uint8), g[None].astype(np.uint8))[0]),
            float(miou_per_image(topk_decision(p)[None], g[None].astype(np.uint8))[0]),
            float(miou_per_image(base[None].astype(np.uint8), g[None].astype(np.uint8))[0])))
    rep, tot = {}, []
    for k, r in rows.items():
        r = np.array(r); tot.append(r)
        rep[k] = dict(n=len(r), auc_med=float(np.nanmedian(r[:, 0])), miou_05=float(r[:, 1].mean()),
                      miou_topk=float(r[:, 2].mean()), base=float(r[:, 3].mean()))
    t = np.concatenate(tot)
    rep["tout"] = dict(n=len(t), auc_med=float(np.nanmedian(t[:, 0])), miou_05=float(t[:, 1].mean()),
                       miou_topk=float(t[:, 2].mean()), base=float(t[:, 3].mean()))
    return rep


def fmt(rep):
    return " | ".join(f"{k} (n={v['n']}) AUC {v['auc_med']:.3f} mIoU@.5 {v['miou_05']:.3f} top-k {v['miou_topk']:.3f} "
                      f"[base {v['base']:.3f}]" for k, v in rep.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--timm_weights", default=None)
    ap.add_argument("--run", default="v1")
    ap.add_argument("--pairs", default=str(HERE / "pairs"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--keep_last", action="store_true", help="garder le dernier pas plutôt que le meilleur")
    ap.add_argument("--eval_n", type=int, default=0, help="tests : nb d'images réelles évaluées (0 = toutes)")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", default=None, help="k5 : frames_<méthode>.json — vraies images mises au cadre")
    ap.add_argument("--predict_only", action="store_true", help="k5 : pas d'entraînement, prédictions train + test avec --init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_rect", type=float, default=0.0, help="part d'exemples « rectangle » (si générés)")
    ap.add_argument("--init", default=None, help="poids de départ (ex. runs/v1/pixelseal.pt)")
    ap.add_argument("--fold", type=int, default=-1, help="CV : vraies images des 4 autres plis dans l'entraînement, "
                    "validation sur ce pli (sortie runs/<run>/f<k>/)")
    ap.add_argument("--real_frac", type=float, default=0.5, help="part de vraies images dans les lots (avec --fold)")
    ap.add_argument("--pseudo", nargs="*", default=[], help="probabilités test (npz) de modèles qui n'ont PAS vu le pli "
                    "validé -> pseudo-étiquettes (top-k, bande autour de la frontière ignorée)")
    ap.add_argument("--pseudo_frac", type=float, default=0.2, help="part d'images de test pseudo-étiquetées dans les lots")
    ap.add_argument("--pseudo_band", type=int, default=2, help="largeur (cellules) de la bande ignorée")
    ap.add_argument("--p_fill", type=float, default=0.0, help="part des composites dont la zone non tatouée vient d'une autre image")
    ap.add_argument("--p_blur", type=float, default=0.0, help="part des composites full aux trous lissés")
    ap.add_argument("--p_wdeg", type=float, default=0.0, help="A1c : part des composites full dont le trou est "
                    "une version DÉGRADÉE de l'image tatouée (hypothèse non tranchée : en mélange seulement)")
    ap.add_argument("--alpha", type=float, nargs=2, default=(1.0, 1.0), metavar=("LO", "HI"),
                    help="A1b : une fois sur deux, force du tatouage tirée dans [LO, HI] (ex. 0.8 1.2)")
    ap.add_argument("--scale2", action="store_true", help="A3 : deuxième échelle (carte fine 128 au lieu de 64)")
    ap.add_argument("--stem_div", type=int, default=1, help="A3 variante 2 : divise le pas de la tige")
    ap.add_argument("--miou_w", type=float, default=0.5, help="A4b : poids du terme mIoU souple dans la perte")
    ap.add_argument("--synth_n", type=int, default=48, help="A2 : images de la validation synthétique fixe (0 = aucune)")
    ap.add_argument("--synth_floor", type=float, default=0.0, help="A2 : arrêt si le mIoU top-k synthétique est "
                    "sous ce seuil à --synth_floor_step (0 = pas d'arrêt)")
    ap.add_argument("--synth_floor_step", type=int, default=20000)
    ap.add_argument("--max_min", type=float, default=0.0, help="échéance : arrêt propre après N minutes (0 = aucune)")
    ap.add_argument("--budget_min", type=float, default=0.0, help="A2 : durée visée en minutes. Le coût par pas est "
                    "mesuré sur les 200 premiers pas, puis le nombre de pas (et le recuit du pas d'apprentissage) "
                    "est ramené à ce qui tient dans le budget, sans dépasser --steps. 0 = utiliser --steps tel quel.")
    a = ap.parse_args()
    if a.frames:
        load_frames(a.frames)
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    arch = a.arch or default_arch(a.method)
    out = HERE / "runs" / a.run / (f"f{a.fold}" if a.fold >= 0 else "")
    retry(lambda: out.mkdir(parents=True, exist_ok=True), "dossier du run")
    tag = a.method
    model = build_model(a.method, arch, device, a.timm_weights, a.scale2, a.stem_div)
    opt = torch.optim.AdamW([dict(params=model.backbone_params(), lr=a.lr_backbone),
                             dict(params=model.head_params(), lr=a.lr)], weight_decay=1e-4)
    warm = min(300, max(1, a.steps // 10))
    plan = dict(total=a.steps)                        # modifiable : --budget_min ajuste le total après mesure
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, plan["total"] - warm))))
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device))
    gt = load_gt128()
    real, val_names = None, None
    if a.fold >= 0:                                               # mêmes plis que assemble.py et le reste du projet
        allm = load_meta("train")
        folds = stratified_folds([m["method"] for m in allm], 5, 0)
        mine = [(m["image_name"], f) for m, f in zip(allm, folds) if m["method"] == a.method]
        real = [(n, gt[n]) for n, f in mine if f != a.fold]
        val_names = {n for n, f in mine if f == a.fold}
    pseudo = pseudo_labels(a.pseudo, a.pseudo_band) if a.pseudo else None
    if pseudo:
        frac1 = np.median([lab.mean() for _, lab, _ in pseudo]); ign = 1 - np.mean([w.mean() for _, _, w in pseudo])
        print(f"pseudo-étiquettes : {len(pseudo)} images de test, fraction tatouée médiane {frac1:.2f}, "
              f"cellules ignorées {ign:.1%}", flush=True)
    ds = SynthSet(a.method, a.pairs, a.steps * a.bs, a.seed, a.p_rect, real, a.real_frac, pseudo, a.pseudo_frac,
                  a.p_fill, a.p_blur, a.p_wdeg, a.alpha)
    sv = SynthVal(a.method, a.pairs, a.synth_n) if a.synth_n else None
    if sv is not None and not len(sv):
        print("pas de validation synthétique : aucune paire « holdout » dans index.json "
              "(regénérer avec gen_pairs.py --holdout N)", flush=True)
        sv = None
    dl = DataLoader(ds, a.bs, shuffle=False, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    real_tr = RealSet("train", a.method, a.eval_n, val_names)
    print(f"{a.method} arch={arch} | {len(ds.index)} paires | {a.steps} pas × {a.bs} | {len(real_tr)} images réelles "
          f"de validation | {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M paramètres", flush=True)
    log, t0, run_loss = [], time.time(), 0.0
    t200 = t0
    if a.predict_only:
        pr = predict(model, real_tr, device, workers=min(4, a.workers))
        print(f"[VAL RÉEL prédiction seule] {fmt(evaluate(pr, real_tr, gt))}", flush=True)
        dl = []; a.keep_last = True
    best = (-1.0, 0, None, None)
    model.train()
    for step, (y, tgt, wgt) in enumerate(dl):
        y = y.to(device, non_blocking=True).float() / 255; tgt = tgt.to(device, non_blocking=True)
        wgt = wgt.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(y)
        loss = loss_fn(logits.float(), tgt, wgt, a.miou_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        run_loss += loss.item()
        if not math.isfinite(loss.item()):
            raise RuntimeError("perte NaN/Inf")
        if (step + 1) % 100 == 0:
            print(f"pas {step + 1}/{plan['total']} perte {run_loss / 100:.4f} "
                  f"({(time.time() - t0) / (step + 1):.2f}s/pas)", flush=True)
            log.append(dict(step=step + 1, loss=run_loss / 100)); run_loss = 0.0
        if a.budget_min and step + 1 == 200:          # coût par pas mesuré, hors démarrage des workers
            sps = (time.time() - t200) / 100
            n_eval = max(1, a.steps // a.eval_every)
            fit = int((a.budget_min * 60 - n_eval * 45) / max(sps, 1e-6))   # ~45 s par évaluation (218 images)
            plan["total"] = int(np.clip(fit, 1000, a.steps))
            print(f"budget {a.budget_min:.0f} min à {sps:.3f} s/pas -> {plan['total']} pas "
                  f"(plafond --steps {a.steps})", flush=True)
            log.append(dict(step=200, s_par_pas=sps, total=plan["total"]))
        if step + 1 == 100:
            t200 = time.time()
        if (step + 1) % a.eval_every == 0 or step + 1 >= plan["total"]:
            pr = predict(model, real_tr, device, workers=min(4, a.workers))
            rep = evaluate(pr, real_tr, gt)
            print(f"[VAL RÉEL pas {step + 1}] {fmt(rep)}", flush=True)
            entry = dict(step=step + 1, val=rep)
            syn = None
            if sv is not None:
                syn = evaluate_synth(predict(model, sv, device, workers=0), sv)
                print(f"[VAL SYNTH pas {step + 1}] " + " | ".join(
                    f"{k} (n={v['n']}) AUC {v['auc_med']:.3f} top-k {v['miou_topk']:.3f}" for k, v in syn.items()),
                    flush=True)
                entry["synth"] = syn
            log.append(entry)
            score = float(np.mean([rep[k]["auc_med"] for k in ("full", "objet") if k in rep]))
            if score > best[0] and not a.keep_last:                                   # meilleur point d'arrêt (AUC full + objet)
                best = (score, step + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, pr)
                # point de contrôle : si le job est coupé (échéance, nœud repris), les poids existent déjà
                retry(lambda: torch.save(best[2], str(out / f"{tag}_ckpt.pt")), "point de contrôle")
            model.train()
            if (a.synth_floor and syn is not None and step + 1 >= a.synth_floor_step
                    and syn["tout"]["miou_topk"] < a.synth_floor):
                print(f"ARRÊT : mIoU top-k synthétique {syn['tout']['miou_topk']:.3f} < {a.synth_floor} "
                      f"au pas {step + 1} — la limite est le signal, pas la durée", flush=True)
                break
            if a.max_min and (time.time() - t0) / 60 > a.max_min:
                print(f"ARRÊT : échéance de {a.max_min:.0f} min atteinte au pas {step + 1}", flush=True)
                break
            if step + 1 >= plan["total"]:
                break
    if best[2] is not None and not a.keep_last:
        print(f"meilleur point d'arrêt : pas {best[1]} (AUC full+objet {best[0]:.3f})", flush=True)
        model.load_state_dict(best[2]); pr = best[3]
    retry(lambda: np.savez_compressed(out / f"{tag}_train.npz", probs=pr,
                                      name=np.array([m["image_name"] for m in real_tr.meta])), "probas train")
    if not a.no_test:
        te = RealSet("test", a.method)
        pt = predict(model, te, device, workers=min(4, a.workers))
        retry(lambda: np.savez_compressed(out / f"{tag}_test.npz", probs=pt,
                                          name=np.array([m["image_name"] for m in te.meta])), "probas test")
    retry(lambda: torch.save(model.state_dict(), str(out / f"{tag}.pt")), "poids")
    retry(lambda: (out / f"{tag}_log.json").write_text(json.dumps(dict(args=vars(a), arch=arch, log=log), indent=1)), "log")
    print(f"FIN {a.method} en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
