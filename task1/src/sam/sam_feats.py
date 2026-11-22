"""Tâche 1, méthode « propositions d'objets » — étape GPU : segments SAM + caractéristiques par segment.

Idée : les zones à trouver ont la forme d'objets (trous des images full, objets transplantés). Au lieu de
décider pixel par pixel, on propose des objets (SAM) puis on teste chaque objet en agrégeant les preuves sur
toute sa surface (le rapport signal/bruit croît comme la racine du nombre de pixels).

Pour chaque image d'un split, sur une part (shard) des images :
  1. SAM ViT-H, générateur automatique de masques -> segments candidats (contours d'objets).
  2. Cartes de preuve en 512 (deux bandes de fréquence : y − flou σ1 ; flou σ1 − flou σ3) :
       r = résidu de l'image
       c = embed(y, m) − embed(y, m̄)   contraste de message : ne garde que la part du motif qui dépend du
                                        message connu (la part liée au contenu, qui corrèle avec la texture
                                        même dans les trous, s'annule)
       d = embed(y, m) − y              motif prédit (statistique de l'essai B, pour comparaison)
  3. Caractéristiques par segment : corrélations r·c et r·d dans le segment, dans un anneau autour et sur le
     reste de l'image ; énergies ; géométrie ; probabilité moyenne du segmenteur U-Net (hors-pli pour le
     train) ; log-vraisemblance du message en décodant le recadrage du segment (3 échelles).
  4. Sortie : out/<split>/<image>.npz (masques 512 compressés, caractéristiques ; fraction tatouée vraie
     pour le train).

  python sam_feats.py --split train --shard 0/4
Smoke test (nœud de login, CPU, 1 image par méthode) : --per_method 1 --device cpu --pps 8 --out_dir out/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
if (HERE / "pylib").exists():                      # segment_anything / sklearn installés à part (pas le .venv partagé)
    sys.path.insert(0, str(HERE / "pylib"))


def _find_task_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "starter_kit" / "wmloc").exists():
            return p
    raise SystemExit("starter_kit introuvable au-dessus de " + str(HERE))


ROOT = _find_task_root()
KIT = ROOT / "starter_kit"
import os
RELEASE = Path(os.environ.get("T1_DATA", ROOT)) / "release"
sys.path.insert(0, str(KIT))
P512 = 512 * 512
FACTORS = (1.0, 1.5, 2.2)                           # recadrage décodé = boîte du segment agrandie de ce facteur


def load_seg_probs(split: str, run_dir: Path) -> dict:
    """Probabilités 128×128 du segmenteur U-Net de Paulin : hors-pli (train) ou moyenne des 5 plis (test)."""
    files = sorted(run_dir.glob("oof_fold*.npz" if split == "train" else "test_fold*.npz"))
    acc = {}
    for f in files:
        z = np.load(f)
        for n, p in zip(z["name"].tolist(), z["probs"]):
            acc.setdefault(n, []).append(p.astype(np.float32))
    return {n: np.mean(v, 0) for n, v in acc.items()}


def load_global_ll(split: str) -> dict:
    p = RELEASE / f"signals_{split}.npz"
    if not p.exists():
        return {}
    s = np.load(p)
    return dict(zip(s["name"].tolist(), s["global_ll"].tolist()))


def bands(x: torch.Tensor):
    """x : 3×H×W float -> (haute fréquence, bande moyenne)."""
    from torchvision.transforms.functional import gaussian_blur
    g1 = gaussian_blur(x[None], [7, 7], [1.0, 1.0])[0]
    g3 = gaussian_blur(x[None], [19, 19], [3.0, 3.0])[0]
    return x - g1, g1 - g3


@torch.no_grad()
def crop_ll(wm, y01: torch.Tensor, boxes: np.ndarray, bits: np.ndarray, bs: int = 32) -> np.ndarray:
    """Log-vraisemblance moyenne du message en décodant chaque recadrage (x1, y1, x2, y2), redimensionné à la
    résolution de travail du décodeur."""
    from torchvision.ops import roi_align
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    R = int(wm.resolution)
    bt = torch.as_tensor(np.c_[np.zeros(len(boxes)), boxes], dtype=torch.float32, device=y01.device)
    out = []
    for i in range(0, len(bt), bs):
        crops = roi_align(y01[None], bt[i:i + bs], output_size=(R, R), spatial_scale=1.0, sampling_ratio=2,
                          aligned=True).clamp(0, 1)
        lg = wm.logits(crops).float()
        sign = torch.as_tensor(2.0 * bits - 1.0, dtype=torch.float32, device=lg.device)
        out.append(F.logsigmoid(lg * sign).mean(1).cpu())
    return torch.cat(out).numpy()


def image_features(img, bits, method, wm, anns, seg_prob, gt512, gll, dev, a):
    """Caractéristiques des S segments d'une image. Retourne (F [S, nf], noms, frac_wm [S] ou None)."""
    S = len(anns)
    M = np.stack([d["segmentation"] for d in anns]).astype(bool)                      # S × 512 × 512
    y = img.astype(np.float32)
    e1 = wm.embed(img, bits).astype(np.float32)
    e0 = wm.embed(img, (1 - bits).astype(np.uint8)).astype(np.float32)
    t3 = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).to(dev)
    Yh, Yb = bands(t3(y)); Ch, Cb = bands(t3(e1 - e0)); Dh, Db = bands(t3(e1 - y))
    prod = lambda u, v: (u * v).sum(0).reshape(-1)
    cols = {"ch_rs": prod(Yh, Ch), "ch_ss": prod(Ch, Ch), "h_rr": prod(Yh, Yh),
            "cb_rs": prod(Yb, Cb), "cb_ss": prod(Cb, Cb), "b_rr": prod(Yb, Yb),
            "dh_rs": prod(Yh, Dh), "dh_ss": prod(Dh, Dh), "db_rs": prod(Yb, Db), "db_ss": prod(Db, Db)}
    if seg_prob is not None:
        cols["seg"] = torch.from_numpy(np.repeat(np.repeat(seg_prob, 4, 0), 4, 1)).to(dev).reshape(-1).float()
    if gt512 is not None:
        cols["gt"] = torch.from_numpy(gt512.astype(np.float32)).to(dev).reshape(-1)
    keys = list(cols)
    V = torch.stack([cols[k] for k in keys], 1)                                       # P × nv
    Mt = torch.from_numpy(M.reshape(S, -1)).to(dev).float()                           # S × P
    k = a.ring
    Dl = F.max_pool2d(Mt.view(S, 1, 512, 512), 2 * k + 1, 1, k).view(S, -1)
    Rg = (Dl - Mt).clamp(0, 1)
    In, Ring, Tot = Mt @ V, Rg @ V, V.sum(0, keepdim=True)
    Out = Tot - In
    col = {kk: i for i, kk in enumerate(keys)}
    n_in = Mt.sum(1).clamp_min(1); n_ring = Rg.sum(1).clamp_min(1); n_out = (P512 - Mt.sum(1)).clamp_min(1)
    eps = 1e-6
    feats = {}
    for sig in ("c", "d"):
        for b in ("h", "b"):
            rs, ss, rr = col[f"{sig}{b}_rs"], col[f"{sig}{b}_ss"], col[f"{b}_rr"]
            rho = {}
            for reg, A in (("in", In), ("ring", Ring), ("out", Out)):
                rho[reg] = A[:, rs] / torch.sqrt(A[:, rr] * A[:, ss] + eps)
                feats[f"rho_{sig}{b}_{reg}"] = rho[reg]
            feats[f"z_{sig}{b}_in"] = rho["in"] * torch.sqrt(n_in)
            feats[f"drho_{sig}{b}_ring"] = rho["in"] - rho["ring"]
            feats[f"drho_{sig}{b}_out"] = rho["in"] - rho["out"]
    for name, c in (("rh", "h_rr"), ("rb", "b_rr"), ("ch", "ch_ss"), ("cb", "cb_ss")):
        img_mean = Tot[0, col[c]] / P512 + eps
        feats[f"e_{name}_in"] = (In[:, col[c]] / n_in) / img_mean
        feats[f"e_{name}_ring"] = (Ring[:, col[c]] / n_ring) / img_mean
    if "seg" in col:
        feats["p_in"] = In[:, col["seg"]] / n_in
        feats["p_ring"] = Ring[:, col["seg"]] / n_ring
        feats["p_out"] = Out[:, col["seg"]] / n_out
        feats["p_img"] = (Tot[0, col["seg"]] / P512).expand(S)
    frac = (In[:, col["gt"]] / n_in).cpu().numpy() if "gt" in col else None
    feats = {kk: v.float().cpu().numpy() for kk, v in feats.items()}
    # géométrie et qualité SAM
    bb = np.array([d["bbox"] for d in anns], np.float32)                              # x, y, w, h
    area = M.reshape(S, -1).sum(1).astype(np.float32)
    feats["area_frac"] = area / P512
    feats["ring_frac"] = Rg.sum(1).cpu().numpy() / P512
    feats["bbox_fill"] = area / np.maximum(bb[:, 2] * bb[:, 3], 1)
    feats["aspect"] = np.maximum(bb[:, 2], bb[:, 3]) / np.maximum(np.minimum(bb[:, 2], bb[:, 3]), 1)
    feats["border"] = ((bb[:, 0] <= 1) | (bb[:, 1] <= 1) | (bb[:, 0] + bb[:, 2] >= 511) |
                       (bb[:, 1] + bb[:, 3] >= 511)).astype(np.float32)
    feats["sam_iou"] = np.array([d["predicted_iou"] for d in anns], np.float32)
    feats["sam_stab"] = np.array([d["stability_score"] for d in anns], np.float32)
    feats["n_segs"] = np.full(S, S, np.float32)
    feats["global_ll"] = np.full(S, np.nan if gll is None else gll, np.float32)
    # décodage des recadrages (segments assez grands, les plus grands d'abord)
    y01 = t3(y) / 255.0
    elig = [i for i in np.argsort(-area) if area[i] >= a.min_dec_area][:a.max_dec]
    for f in FACTORS:
        v = np.full(S, np.nan, np.float32)
        if elig:
            cx = bb[elig, 0] + bb[elig, 2] / 2; cy = bb[elig, 1] + bb[elig, 3] / 2
            hw = np.maximum(bb[elig, 2] * f, 8) / 2; hh = np.maximum(bb[elig, 3] * f, 8) / 2
            boxes = np.stack([np.clip(cx - hw, 0, 512), np.clip(cy - hh, 0, 512),
                              np.clip(cx + hw, 0, 512), np.clip(cy + hh, 0, 512)], 1)
            v[elig] = crop_ll(wm, y01, boxes, bits)
        feats[f"ll_f{f:g}"] = v
    names = sorted(feats)
    return np.stack([feats[n] for n in names], 1).astype(np.float32), names, frac, M, bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--per_method", type=int, default=0, help="tests : N premières images de chaque méthode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=str(HERE / "weights" / "sam_vit_h_4b8939.pth"))
    ap.add_argument("--seg_run", default=str(ROOT / "paulin" / "seg" / "runs" / "unet_a"))
    ap.add_argument("--out_dir", default=str(HERE / "out"))
    ap.add_argument("--pps", type=int, default=32, help="SAM points_per_side")
    ap.add_argument("--pred_iou", type=float, default=0.80)
    ap.add_argument("--stab", type=float, default=0.85)
    ap.add_argument("--crop_layers", type=int, default=0)
    ap.add_argument("--min_area", type=int, default=64)
    ap.add_argument("--max_segs", type=int, default=150)
    ap.add_argument("--ring", type=int, default=12, help="largeur de l'anneau autour du segment (px)")
    ap.add_argument("--min_dec_area", type=int, default=400)
    ap.add_argument("--max_dec", type=int, default=80)
    a = ap.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from wmloc.watermarks import load_watermark

    dev = torch.device(a.device)
    meta = json.loads((RELEASE / a.split / "metadata.json").read_text())
    if a.per_method:
        meta = [m for meth in ("pixelseal", "trustmark", "stegastamp")
                for m in [x for x in meta if x["method"] == meth][:a.per_method]]
    k, n = map(int, a.shard.split("/"))
    meta = [m for i, m in enumerate(meta) if i % n == k]
    out = Path(a.out_dir) / a.split
    out.mkdir(parents=True, exist_ok=True)
    seg_probs = load_seg_probs(a.split, Path(a.seg_run)) if Path(a.seg_run).exists() else {}
    gll = load_global_ll(a.split)
    print(f"{a.split} part {a.shard} : {len(meta)} images ; probas segmenteur : {len(seg_probs)} ; "
          f"global_ll : {len(gll)} ; device {dev}", flush=True)

    sam = sam_model_registry["vit_h"](checkpoint=a.weights).to(dev).eval()
    amg = SamAutomaticMaskGenerator(sam, points_per_side=a.pps, pred_iou_thresh=a.pred_iou,
                                    stability_score_thresh=a.stab, crop_n_layers=a.crop_layers,
                                    min_mask_region_area=a.min_area, points_per_batch=64)
    wms = {}
    mdir = KIT / "models"
    t_start = time.time()
    for i, m in enumerate(meta):
        name, method = m["image_name"], m["method"]
        dst = out / f"{name}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        img = np.asarray(Image.open(RELEASE / a.split / "images" / f"{name}.png").convert("RGB"))
        bits = np.array([int(c) for c in m["message"]], dtype=np.uint8)
        with torch.no_grad():
            anns = amg.generate(img)
        anns = sorted(anns, key=lambda d: -d["predicted_iou"])[:a.max_segs]
        t_sam = time.time() - t0
        gt512 = None
        if a.split == "train":
            gt512 = (np.asarray(Image.open(RELEASE / "train" / "masks" / f"{name}.png")) > 127).astype(np.uint8)
        if not anns:
            np.savez_compressed(dst, masks=np.zeros((0, P512 // 8), np.uint8), feats=np.zeros((0, 0), np.float32),
                                feat_names=np.array([]), frac_wm=np.zeros(0, np.float32), boxes=np.zeros((0, 4)),
                                method=method, t_sam=t_sam)
            print(f"[{i + 1}/{len(meta)}] {name} : aucun segment", flush=True)
            continue
        if method not in wms:
            wms[method] = load_watermark(method, mdir, str(dev))
        Fm, fnames, frac, M, bb = image_features(img, bits, method, wms[method], anns, seg_probs.get(name),
                                                 gt512, gll.get(name), dev, a)
        np.savez_compressed(dst, masks=np.packbits(M.reshape(len(anns), -1), axis=1), feats=Fm,
                            feat_names=np.array(fnames), frac_wm=(frac if frac is not None else np.zeros(0)),
                            boxes=bb, method=method, t_sam=t_sam, t_all=time.time() - t0)
        el = time.time() - t_start
        print(f"[{i + 1}/{len(meta)}] {name} ({method}) : {len(anns)} segments, SAM {t_sam:.1f}s, "
              f"total {time.time() - t0:.1f}s ; reste ~{el / (i + 1) * (len(meta) - i - 1) / 60:.1f} min", flush=True)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
