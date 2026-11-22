"""Recherche du rectangle marqué par décodage de recadrages (piste « outpainting », GPU).

Constat (diag_crop.py, diag_tolerance.py) : décoder le recadrage EXACT de la zone marquée relit le message
(TrustMark ≈ 1,00, StegaStamp ≈ 0,97), alors que l'image entière ou un cadre décalé donne ≈ 0,5.
TrustMark lit encore le message dès IoU(cadre, vrai) ≥ 0,5-0,7 ; StegaStamp demande ≥ 0,9.

Algorithme, pour une image :
  1. balayage grossier : toutes les fenêtres d'une grille multi-échelle (côtés en progression géométrique,
     pas = `stride` × côté), recadrées et redimensionnées en lot sur GPU (roi_align, échantillonnage
     adaptatif ≈ réduction par moyenne) ; score = log-vraisemblance moyenne du message connu ;
  2. affinage des `topk` meilleures : montée bord par bord (pas 1/8 du côté puis de plus en plus fins),
     score recalculé avec le décodeur officiel sur le vrai recadrage (wm.logits) ;
  3. sortie : meilleur cadre, sa log-vraisemblance et sa précision par bit (décision faite ailleurs, seuil
     calibré sur le train).

    python rect_search.py --split train --method trustmark --limit 20 --out out_rect_tm.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

HERE = Path(__file__).resolve().parent


def find_kit():
    for p in (HERE.parent / "starter_kit", HERE / "starter_kit"):
        if (p / "wmloc").exists():
            return p
    raise SystemExit("starter_kit introuvable")


KIT = find_kit()
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}


def batch_logits(wm, method, x):
    """x : B x 3 x R x R dans [0,1], déjà à la résolution de travail du décodeur -> B x nbits."""
    if method == "pixelseal":
        return wm.det(x)[:, 1:].float()
    if method == "trustmark":
        return wm.net(x * 2 - 1).float()
    if getattr(wm, "torch_dec", None) is not None:      # décodeur StegaStamp converti en PyTorch (reste sur GPU)
        return wm.torch_dec(x.clamp(0, 1).permute(0, 2, 3, 1).contiguous()).float()
    xs = x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return torch.from_numpy(wm._run(xs, [wm.LOGITS])[0]).float().to(x.device)


def stegastamp_torch_decoder(models_dir, device):
    """Extrait du graphe ONNX le sous-graphe du DÉCODEUR (image -> logits) et le convertit en PyTorch
    (onnx2torch). Vérifié en local : mêmes bits que onnxruntime, écart max des logits 4e-5."""
    import onnx
    from onnx.utils import Extractor
    from onnx2torch import convert
    dst = Path(models_dir) / "stegastamp" / "stega_decoder_only.onnx"
    if not dst.exists():
        m = onnx.load(str(Path(models_dir) / "stegastamp" / "stega_stamp_logits.onnx"))
        onnx.save(Extractor(m).extract_model(["image"], ["gen_decoder/decoder_out/BiasAdd:0"]), str(dst))
    return convert(onnx.load(str(dst))).eval().to(device)


def grid_boxes(size=512, smin=24, ratio=1.35, stride=0.2, max_aspect=16.0):
    sides = []
    s = smin
    while s <= size:
        sides.append(int(round(s))); s *= ratio
    if sides[-1] != size:
        sides.append(size)
    boxes = []
    for h in sides:
        for w in sides:
            if max(h, w) / min(h, w) > max_aspect:
                continue
            sy, sx = max(2, int(stride * h)), max(2, int(stride * w))
            ys = list(range(0, size - h + 1, sy)); xs = list(range(0, size - w + 1, sx))
            if ys[-1] != size - h: ys.append(size - h)
            if xs[-1] != size - w: xs.append(size - w)
            boxes += [(y, y + h, x, x + w) for y in ys for x in xs]
    return np.array(boxes, dtype=np.int64)   # y0, y1, x0, x1


def coarse_scores(wm, method, img_t, sign, boxes, bs, amp=False):
    """img_t : 1 x 3 x 512 x 512 (device). Renvoie la log-vraisemblance moyenne de chaque fenêtre.
    amp : demi-précision (fp16) pour les décodeurs PyTorch (pixelseal, trustmark) ; le score sert
    seulement à classer les fenêtres, le score final est recalculé en fp32 (exact_score)."""
    R = RES[method]
    out = []
    use_amp = amp and method != "stegastamp" and img_t.is_cuda
    for k in range(0, len(boxes), bs):
        b = torch.as_tensor(boxes[k:k + bs], dtype=torch.float32, device=img_t.device)
        rois = torch.cat([torch.zeros(len(b), 1, device=b.device), b[:, [2, 0, 3, 1]]], 1)  # idx, x0, y0, x1, y1
        crops = roi_align(img_t, rois, output_size=(R, R), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            lg = batch_logits(wm, method, crops)
        out.append(F.logsigmoid(lg.float() * sign[None]).mean(1).cpu())
    return torch.cat(out).numpy()


def exact_score(wm, img_np, sign, box):
    y0, y1, x0, x1 = box
    x = torch.from_numpy(np.ascontiguousarray(img_np[y0:y1, x0:x1])).permute(2, 0, 1)[None].float() / 255
    lg = wm.logits(x)[0].float().cpu()
    return float(F.logsigmoid(lg * sign.cpu()).mean()), float(((lg > 0).float() * 2 - 1 == sign.cpu()).float().mean())


def refine(wm, img_np, sign, box, min_side=8):
    best = tuple(int(v) for v in box)
    best_ll, best_acc = exact_score(wm, img_np, sign, best)
    h, w = best[1] - best[0], best[3] - best[2]
    for frac in (1 / 8, 1 / 16, 1 / 32, 1 / 64):
        steps = (max(1, int(h * frac)), max(1, int(h * frac)), max(1, int(w * frac)), max(1, int(w * frac)))
        improved = True
        while improved:
            improved = False
            for e in range(4):
                for d in (-steps[e], steps[e]):
                    c = list(best); c[e] += d
                    c = [min(max(c[0], 0), 512), min(max(c[1], 0), 512), min(max(c[2], 0), 512), min(max(c[3], 0), 512)]
                    if c[1] - c[0] < min_side or c[3] - c[2] < min_side:
                        continue
                    ll, acc = exact_score(wm, img_np, sign, tuple(c))
                    if ll > best_ll + 1e-4:
                        best, best_ll, best_acc, improved = tuple(c), ll, acc, True
    return best, best_ll, best_acc


def plateau_center(wm, img_np, sign, box, acc_ref, tol=0.03, min_side=8):
    """Le score sature (précision ≈ 1) sur toute une plage de cadres autour du vrai rectangle : la montée
    s'arrête n'importe où sur ce plateau. Pour chaque bord, on mesure l'intervalle où la précision reste
    ≥ acc_ref − tol (pas de 1/64 du côté, jusqu'à ±1/3 du côté), et on place le bord au milieu."""
    cur = list(box)
    for e in range(4):
        side = (cur[1] - cur[0]) if e < 2 else (cur[3] - cur[2])
        step = max(1, side // 64)
        lim = {}
        for direction in (-1, 1):
            pos = cur[e]
            for _ in range(max(1, side // 3) // step):
                c = list(cur); c[e] = pos + direction * step
                c[e] = min(max(c[e], 0), 512)
                if c[1] - c[0] < min_side or c[3] - c[2] < min_side or c[e] == pos:
                    break
                _, acc = exact_score(wm, img_np, sign, tuple(c))
                if acc < acc_ref - tol:
                    break
                pos = c[e]
            lim[direction] = pos
        cur[e] = int(round((lim[-1] + lim[1]) / 2))
    ll, acc = exact_score(wm, img_np, sign, tuple(cur))
    return tuple(cur), ll, acc


def box_iou(bb, gt):
    iy = max(0, min(bb[1], gt[1]) - max(bb[0], gt[0])); ix = max(0, min(bb[3], gt[3]) - max(bb[2], gt[2]))
    inter = iy * ix
    return inter / ((bb[1] - bb[0]) * (bb[3] - bb[2]) + (gt[1] - gt[0]) * (gt[3] - gt[2]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(KIT.parent / "release"))
    ap.add_argument("--models", default=str(KIT / "models"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--method", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", default="", help="fichier texte : une image par ligne (sinon non-full du train)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--stride", type=float, default=0.2)
    ap.add_argument("--ratio", type=float, default=1.35)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amp", action="store_true", help="balayage grossier en fp16 (pixelseal, trustmark)")
    ap.add_argument("--shard", default="0/1", help="i/n : ne traite que les images d'indice ≡ i (mod n)")
    ap.add_argument("--ss_torch", action="store_true", help="StegaStamp : décodeur converti en PyTorch (balayage)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    ds = LocalisationDataset(a.data, a.split)
    if a.names:
        wanted = set(Path(a.names).read_text().split())
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == a.method and m["image_name"] in wanted]
    else:
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == a.method and not m.get("full_watermark", False)]
    if a.limit:
        idx = idx[:a.limit]
    si, sn = map(int, a.shard.split("/"))
    idx = idx[si::sn]
    wm = load_watermark(a.method, a.models, a.device)
    if a.method == "stegastamp" and a.ss_torch:
        # --ss_torch needs `pip install onnx2torch`, left out of requirements.txt on purpose: the submitted
        # pipeline only searches TrustMark rectangles (StegaStamp ones measured -0.0002, see README section 5).
        wm.torch_dec = stegastamp_torch_decoder(a.models, a.device)
    boxes = grid_boxes(stride=a.stride, ratio=a.ratio)
    print(f"{a.method} : {len(idx)} images, {len(boxes)} fenêtres par image, device {a.device}", flush=True)

    res = []
    for n, i in enumerate(idx):
        t0 = time.time()
        it = ds[i]
        img_t = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().div(255).to(a.device)
        sign = (torch.as_tensor(it["message"], dtype=torch.float32, device=a.device) * 2 - 1)
        sc = coarse_scores(wm, a.method, img_t, sign, boxes, a.bs, a.amp)
        t1 = time.time()
        order = np.argsort(-sc)
        picked = []
        for j in order:   # top-k fenêtres non redondantes (centres éloignés)
            b = boxes[j]
            if all(abs((b[0] + b[1]) - (p[0] + p[1])) + abs((b[2] + b[3]) - (p[2] + p[3])) > 0.5 * ((b[1] - b[0]) + (b[3] - b[2])) for p in picked):
                picked.append(b)
            if len(picked) >= a.topk:
                break
        cands = [refine(wm, it["image"], sign, b) for b in picked]
        best = max(cands, key=lambda c: c[1])
        mid = plateau_center(wm, it["image"], sign, best[0], best[2]) if best[2] >= 0.8 else best
        r = {"image_name": it["image_name"], "method": a.method,
             "box": list(map(int, mid[0])), "ll": mid[1], "acc": mid[2],
             "box_climb": list(map(int, best[0])), "ll_climb": best[1], "acc_climb": best[2],
             "coarse_best_ll": float(sc[order[0]]), "coarse_best_box": list(map(int, boxes[order[0]])),
             "t_coarse": t1 - t0, "t_total": time.time() - t0}
        if "mask" in it:
            ys, xs = np.nonzero(it["mask"])
            gt = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
            r["gt_box"] = gt
            r["gt_is_rect"] = bool(it["mask"][gt[0]:gt[1], gt[2]:gt[3]].all())
            r["iou_gt_box"] = box_iou(mid[0], gt)
            r["iou_gt_box_climb"] = box_iou(best[0], gt)
        res.append(r)
        print(f"[{n + 1}/{len(idx)}] {r['image_name']} acc {r['acc']:.2f} ll {r['ll']:+.3f} "
              f"iou {r.get('iou_gt_box', float('nan')):.2f} (montée {r.get('iou_gt_box_climb', float('nan')):.2f}) "
              f"rect {r.get('gt_is_rect')} ({r['t_coarse']:.1f}+{r['t_total'] - r['t_coarse']:.1f} s)", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("écrit", a.out)


if __name__ == "__main__":
    main()
