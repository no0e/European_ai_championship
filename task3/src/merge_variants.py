"""Merge the shards written by gen_variants.py into the single file read by k3.py ($K3_SYN, default
$T3_WORK/k5/syn_s0.5.npz). Added for this repository: the merge was a one-off command during the hackathon.
Usage: python src/merge_variants.py k5/var_s0.5_0.npz k5/var_s0.5_1.npz ... --out k5/syn_s0.5.npz"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
parts = [np.load(f) for f in a.shards]
images = np.concatenate([p["images"] for p in parts])
src = np.concatenate([p["src"] for p in parts])
counts = np.bincount(src)
assert counts.min() == counts.max(), "every forget image must have the same number of variants"
np.savez(a.out, images=images, src=src)
print("OK", a.out, images.shape, f"{counts.max()} variants per image")
