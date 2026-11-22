"""Détecteur dense — étape 1 : paires (image propre x, image entièrement tatouée w = embed(x, m)).

Images propres, du même domaine que l'épreuve : les images « non-full » du train (vérité) et du test (régime
prédit), propres à ~96 % ; plus quelques recadrages DIV2K. Chaque image sert `--reps` fois (retournement
horizontal aléatoire AVANT tatouage, message aléatoire). Le tatouage est fait à l'échelle de l'image entière,
comme pour les images full et les objets transplantés de l'épreuve.

  python gen_pairs.py --method pixelseal [--reps 2] [--limit 0] [--device cuda]
Sortie : pairs/<méthode>/x.npy, w.npy (uint8 [N,512,512,3]), r.npy (recadrages tatoués, à plat), index.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image

from common import HERE, KIT, METHODS, RELEASE, ROOT, global_ll, is_rect, load_img, load_meta, regime_thresholds


def train_rect_sizes(method):
    """(hauteur, largeur) des zones rectangulaires du train (outpainting), d'abord celles de la méthode."""
    own, others = [], []
    for m in load_meta("train"):
        if m["full_watermark"]:
            continue
        mk = np.asarray(Image.open(RELEASE / "train" / "masks" / f"{m['image_name']}.png").convert("L")) > 127
        if is_rect(mk):
            ys, xs = np.nonzero(mk)
            (own if m["method"] == method else others).append((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    return own + others if len(own) < 10 else own


def base_pool(method, with_full=False):
    """[(split, nom)] : images non-full des DEUX AUTRES méthodes (train : vérité ; test : régime prédit).

    Surtout pas celles de la méthode entraînée : elles contiennent une vraie zone tatouée par cette méthode, que
    le composite étiquetterait « non tatouée » (c'est ce qui faisait baisser la v1 sur les vraies images au fil
    de l'entraînement). Le tatouage d'une autre méthode, lui, est bien un négatif pour ce détecteur.

    with_full (A2) : ajoute aussi les images FULL des deux autres méthodes, ce qui double à peu près le
    réservoir. Même raisonnement : entièrement tatouées par une AUTRE méthode, donc négatives pour celle-ci.
    C'est le réservoir que `diag_a1.py` utilisait comme « bases jamais vues » : une fois l'option activée,
    ce diagnostic n'est plus valide, et c'est la validation synthétique sur les images « holdout » qui le
    remplace."""
    pool = [("train", m["image_name"]) for m in load_meta("train")
            if m["method"] != method and (with_full or not m["full_watermark"])]
    thr, ll = regime_thresholds(), global_ll("test")
    pool += [("test", m["image_name"]) for m in load_meta("test")
             if m["method"] != method and m["image_name"] in ll
             and (with_full or ll[m["image_name"]] <= thr[m["method"]])]
    return pool


def div2k_crops(rng, n_per_image=3):
    d = ROOT / "clean" / "DIV2K_valid_HR"
    out = []
    for p in sorted(d.glob("*.png")) if d.exists() else []:
        im = Image.open(p).convert("RGB")
        s = 768 / min(im.size)
        im = im.resize((max(512, round(im.size[0] * s)), max(512, round(im.size[1] * s))), Image.Resampling.BICUBIC)
        a = np.asarray(im)
        for k in range(n_per_image):
            y0 = int(rng.integers(0, a.shape[0] - 511)); x0 = int(rng.integers(0, a.shape[1] - 511))
            out.append((f"div2k_{p.stem}_{k}", a[y0:y0 + 512, x0:x0 + 512]))
    return out


def add_rects(wm, x, sizes, k, rng, store):
    """k recadrages rectangulaires de x, tatoués seuls (comme l'outpainting de l'épreuve).
    Retourne [[y0, y1, x0, x1, décalage dans r.npy], ...] ; les pixels vont dans `store`."""
    boxes = []
    for _ in range(k):
        h0, w0 = sizes[int(rng.integers(len(sizes)))]
        h = int(np.clip(h0 * rng.uniform(0.7, 1.4), 16, 512)); w = int(np.clip(w0 * rng.uniform(0.7, 1.4), 16, 512))
        y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
        bits = rng.integers(0, 2, wm.nbits).astype(np.uint8)
        crop = wm.embed(np.ascontiguousarray(x[y0:y0 + h, x0:x0 + w]), bits)
        boxes.append([y0, y0 + h, x0, x0 + w, store["n"]])
        store["chunks"].append(crop.reshape(-1)); store["n"] += crop.size
    return boxes


def retry(fn, what, n=40, wait=20):
    """Le quota du projet partagé sature par moments (« Disk quota exceeded ») : on réessaie."""
    for k in range(n):
        try:
            return fn()
        except (OSError, RuntimeError) as e:
            print(f"écriture refusée ({what}) : {e} ; nouvel essai dans {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"écriture impossible : {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_div2k", action="store_true")
    ap.add_argument("--div2k", type=int, default=4, help="recadrages par image DIV2K")
    ap.add_argument("--rects", type=int, default=0, help="nb de variantes « rectangle » par paire (recadrage tatoué seul)")
    ap.add_argument("--with_full", action="store_true", help="A2 : ajouter les images full des deux autres méthodes")
    ap.add_argument("--holdout", type=int, default=0, help="A2 : images de base tenues à l'écart de l'entraînement, "
                    "pour la validation synthétique fixe de train_dense.py")
    ap.add_argument("--frame", type=float, default=0.0, help="k5 : part des paires tatouées sur un cadre PLUS GRAND que "
                    "l'image (mesuré sur le réel : l'image publiée est un recadrage du cadre tatoué, facteur 1,05-1,15)")
    ap.add_argument("--frame_max", type=float, default=1.30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "pairs"))
    a = ap.parse_args()
    from pathlib import Path
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, KIT / "models", a.device)
    rng = np.random.default_rng(METHODS.index(a.method) + 1)
    out = Path(a.out) / a.method
    retry(lambda: out.mkdir(parents=True, exist_ok=True), "dossier des paires")
    items = [(f"{s}_{n}", s, n, None) for s, n in base_pool(a.method, a.with_full)]
    n_real = len(items)
    if not a.no_div2k:
        items += [(i, "div2k", i, arr) for i, arr in div2k_crops(np.random.default_rng(0), a.div2k)]
    if a.limit:
        items = items[:a.limit]
        n_real = min(n_real, a.limit)
    hold = set()
    if a.holdout:                                      # tirage déterministe, parmi les vraies images seulement
        r = np.random.default_rng(12345)
        hold = {items[i][0] for i in r.permutation(n_real)[:a.holdout]}
    gb = len(items) * a.reps * 512 * 512 * 3 * 2 / 1e9
    print(f"{a.method} : {len(items)} images de base ({n_real} réelles) × {a.reps} = {len(items) * a.reps} paires, "
          f"{len(hold)} tenues à l'écart | x.npy + w.npy ≈ {gb:.1f} Go", flush=True)
    index, t0 = [], time.time()
    sizes = train_rect_sizes(a.method) if a.rects else []
    N = len(items) * a.reps
    # peu de gros fichiers plutôt que des milliers de PNG (quota de fichiers du projet partagé)
    X = retry(lambda: np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3)), "x.npy")
    W = retry(lambda: np.lib.format.open_memmap(out / "w.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3)), "w.npy")
    store = dict(chunks=[], n=0)
    for j, (iid, split, name, arr) in enumerate(items):
        x0 = arr if arr is not None else load_img(split, name)
        for r in range(a.reps):
            flip = bool(rng.random() < 0.5)
            x = np.ascontiguousarray(x0[:, ::-1] if flip else x0)
            bits = rng.integers(0, 2, wm.nbits).astype(np.uint8)
            fr = None
            if rng.random() < a.frame:                 # cadre tatoué plus grand : complété par réflexion, puis recadré
                sx = float(rng.uniform(1.02, a.frame_max))
                sy = sx if rng.random() < 0.9 else float(rng.uniform(1.02, a.frame_max))
                Fw, Fh = int(round(512 * sx)), int(round(512 * sy))
                ox, oy = int(rng.integers(0, Fw - 512 + 1)), int(rng.integers(0, Fh - 512 + 1))
                can = np.pad(x, ((oy, Fh - 512 - oy), (ox, Fw - 512 - ox), (0, 0)), mode="reflect")
                w = np.ascontiguousarray(wm.embed(np.ascontiguousarray(can), bits)[oy:oy + 512, ox:ox + 512])
                fr = [Fw, Fh, ox, oy]
            else:
                w = wm.embed(x, bits)
            k = len(index)
            X[k] = x; W[k] = w
            index.append(dict(id=f"{iid}_{r}", idx=k, split=split, name=name, flip=flip))
            if fr:
                index[-1]["frame"] = fr
            if iid in hold:
                index[-1]["holdout"] = True
            if a.rects:
                index[-1]["rects"] = add_rects(wm, x, sizes, a.rects, rng, store)
        if j % 100 == 0:
            d = np.abs(w.astype(np.float32) - x.astype(np.float32))
            print(f"{a.method} {j}/{len(items)} {time.time() - t0:.0f}s | |w−x| moyen {d.mean():.2f} max {d.max():.0f}",
                  flush=True)
    X.flush(); W.flush()
    if store["n"]:
        R = np.concatenate(store["chunks"])
        retry(lambda: np.save(out / "r.npy", R), "r.npy")
    retry(lambda: (out / "index.json").write_text(json.dumps(index)), "index.json")
    print(f"FIN {a.method} : {len(index)} paires en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
