"""Calcule et met en cache, pour chaque image d'un split, les signaux des décodeurs officiels :

  - global_margin : moyenne sur les bits de sign(message) * logit, image entière (le décodeur « lit-il »
                    le message ?) ; global_acc : précision par bit ; global_ll : log-vraisemblance moyenne
                    du message, mean log sigmoid(sign * logit).
  - map8 (pixelseal, trustmark) : carte h x w de la marge signée par position, à partir de
                    features()["per_position_logits"] (logits avant le pooling moyen de la tête).
StegaStamp n'expose que les logits globaux (graphe ONNX) : pas de map8 pour lui ici.

    python task1/src/compute_signals.py --split train [--limit 20] [--device cpu]

Sortie : <data>/signals_<split>.npz (dans release/, ignoré par git).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KIT = HERE / "starter_kit"
sys.path.insert(0, str(KIT))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)[None].float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(KIT.parent / "release"))
    ap.add_argument("--models", default=str(KIT / "models"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="n images max par méthode (0 = toutes)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    ds = LocalisationDataset(a.data, a.split)
    out = {k: [] for k in ("name", "method", "global_margin", "global_acc", "global_ll")}
    maps = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        wm = load_watermark(method, a.models, a.device)
        idx = [i for i, m in enumerate(ds.meta) if m["method"] == method]
        if a.limit:
            idx = idx[:a.limit]
        t0 = time.time()
        for i in idx:
            it = ds[i]
            x = to_tensor(it["image"])
            sign = torch.as_tensor(it["message"], dtype=torch.float32) * 2 - 1
            logit = wm.logits(x)[0].float().cpu()
            sm = sign * logit
            out["name"].append(it["image_name"]); out["method"].append(method)
            out["global_margin"].append(float(sm.mean()))
            out["global_acc"].append(float(((logit > 0).float() * 2 - 1 == sign).float().mean()))
            out["global_ll"].append(float(F.logsigmoid(sm).mean()))
            if method != "stegastamp":
                pp = wm.features(x)["per_position_logits"][0].float().cpu()   # nbits x h x w
                maps[it["image_name"]] = (pp * sign[:, None, None]).mean(0).numpy()
        dt = (time.time() - t0) / max(len(idx), 1)
        print(f"{method}: {len(idx)} images, {dt:.2f} s/image", flush=True)
        del wm

    names = np.array(out["name"])
    shapes = {m.shape for m in maps.values()}
    print("formes des cartes :", shapes)
    path = Path(a.data) / f"signals_{a.split}{'_lim%d' % a.limit if a.limit else ''}.npz"
    np.savez_compressed(path, name=names, method=np.array(out["method"]),
                        global_margin=np.array(out["global_margin"]), global_acc=np.array(out["global_acc"]),
                        global_ll=np.array(out["global_ll"]),
                        map_names=np.array(list(maps)), maps=np.stack(list(maps.values())) if maps else np.zeros(0))
    print("écrit", path)


if __name__ == "__main__":
    main()
