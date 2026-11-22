"""Comparaison APPARIÉE de deux versions sur les mêmes images du train (scores CV par image de assemble.py --save_cv).

Différence par image B − A, moyenne et intervalle bootstrap à 95 %, au total, par méthode et par (méthode, type).
Seules les images présentes dans les deux fichiers sont comparées.

    python compare_cv.py out/cv_v2.npz out/cv_v3.npz
"""
from __future__ import annotations

import sys

import numpy as np


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(fa, fb):
    A, B = np.load(fa), np.load(fb)
    ia = {n: i for i, n in enumerate(A["name"].tolist())}
    common = [n for n in B["name"].tolist() if n in ia]
    ja = np.array([ia[n] for n in common]); jb = np.array([i for i, n in enumerate(B["name"].tolist()) if n in ia])
    sa, sb = A["score"][ja], B["score"][jb]
    meth, kind, fold = B["method"][jb], B["kind"][jb], B["fold"][jb]
    d = sb - sa
    print(f"A = {fa}\nB = {fb}\n{len(common)} images communes")
    print(f"{'groupe':<22}{'n':>4}{'A':>8}{'B':>8}{'B−A':>9}   IC 95 %")
    groups = [("TOUT", np.ones(len(d), bool))]
    for m in np.unique(meth):
        groups.append((m, meth == m))
        groups += [(f"  {m} {k}", (meth == m) & (kind == k)) for k in ("full", "objet", "rect") if ((meth == m) & (kind == k)).any()]
    for g, s in groups:
        mu, lo, hi = boot(d[s])
        print(f"{g:<22}{s.sum():>4}{sa[s].mean():>8.4f}{sb[s].mean():>8.4f}{mu:>+9.4f}   [{lo:+.4f} ; {hi:+.4f}]")
    pa = [sa[fold == f].mean() for f in range(5)]; pb = [sb[fold == f].mean() for f in range(5)]
    print(f"par pli A {np.round(pa, 3).tolist()} (± {np.std(pa, ddof=1):.4f}) ; "
          f"B {np.round(pb, 3).tolist()} (± {np.std(pb, ddof=1):.4f})")
    return boot(d)                                   # (différence moyenne, borne basse, borne haute) sur TOUT


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--gate"]
    mu, lo, hi = main(args[0], args[1])
    if "--gate" in sys.argv:
        # Garde-fou de soumission : la règle de l'équipe est « on ne garde que si l'intervalle est au-dessus
        # de 0 ». Code 0 = gain établi, 1 = non établi. Un script d'envoi doit s'y conformer.
        ok = lo > 0
        print(f"GATE : {'OK' if ok else 'NON'} — B − A = {mu:+.4f} [{lo:+.4f} ; {hi:+.4f}] "
              f"({'borne basse au-dessus de 0' if ok else 'intervalle non concluant : ne pas soumettre'})")
        sys.exit(0 if ok else 1)
