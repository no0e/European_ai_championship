"""Construit anchors.npz : images naturelles publiques (COCO val2017) en uint8 [N,224,224,3],
même format que forget_images.npz. À lancer sur le nœud LOGIN (les nœuds GPU n'ont pas internet).
Usage : python3 prepare_anchors.py <dossier_images> <sortie.npz> [N_max]
"""
import sys, os, glob
import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
files = sorted(glob.glob(os.path.join(src, "*.jpg")))[:n_max]
arr = np.zeros((len(files), 224, 224, 3), dtype=np.uint8)
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)                      # resize côté court à 256 puis center crop 224
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    arr[i] = np.asarray(im.crop((l, t, l + 224, t + 224)))
    if i % 500 == 0:
        print(i, "/", len(files), flush=True)
np.savez(out, images=arr)
print("OK", arr.shape, arr.dtype, "->", out)
