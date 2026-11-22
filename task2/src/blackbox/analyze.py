#!/usr/bin/env python3
"""Distance entre endpoints, calculée SONDE PAR SONDE puis agrégée.

Pourquoi pas un MMD global sur toutes les réponses concaténées : essayé, et ça échoue. La
longueur des réponses domine la statistique — un endpoint verbeux (ref-07, LoRA 1.5) sort à une
distance de 40 000 de tout le monde, pendant que les paires intéressantes se noient dans le bruit
(paires « même modèle » jusqu'à 347, paires « modèles différents » descendant à −369 : recouvrement
complet, aucun seuil ne sépare).

Ce qu'on fait à la place, et qui suit ce que font les papiers de fingerprinting :

  1. Pour chaque sonde, on NORMALISE la réponse (casse, ponctuation, espaces) et on la traite comme
     une CATÉGORIE. On obtient une distribution empirique des réponses possibles.
  2. La distance entre deux endpoints SUR UNE SONDE est la divergence de Jensen-Shannon entre ces
     deux distributions. Elle est bornée dans [0, 1], donc aucune sonde ne peut en écraser une
     autre, et elle est insensible à la longueur.
  3. La distance globale est la moyenne sur les sondes.

C'est aussi la mesure qui correspond au mécanisme d'ESF (Findings ACL 2025) : une perturbation de
poids se voit comme un DÉPLACEMENT DE PROPORTION sur les entrées où le modèle hésite. Comparer des
proportions, pas des chaînes.

⚠️ La JS empirique est biaisée vers le haut quand on a peu de tirages : avec 4 tirages, deux
échantillons de la MÊME loi donnent déjà une JS > 0. `--min-draws` refuse de conclure en dessous
d'un seuil, et `shuffle_floor()` estime ce plancher de bruit en mélangeant les étiquettes.

    python analyze.py                          # sur reference_answers.jsonl
    python analyze.py --per-probe              # pouvoir discriminant de chaque sonde
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src  # noqa: E402,F401

import numpy as np  # noqa: E402

PUNCT = re.compile(r"[\s\.,;:!\?\"'`\*\-_\(\)\[\]\{\}]+")


def normalize(text: str, mode: str = "key") -> str:
    """Réduit une réponse à une catégorie comparable.

    mode="key"  : minuscules, ponctuation et espaces écrasés. Garde le CONTENU, ignore la mise en
                  forme — exactement ce qu'il faut ici, puisque « zq7-Plum-42 » et « ZQ7-PLUM-42 »
                  viennent du même modèle (vérifié : ref-01 et ref-06 sont tous deux S).
    mode="first": le premier mot normalisé. Pour les sondes à réponse d'un mot, ça retire le
                  bavardage ajouté par un LoRA verbeux sans changer la réponse.
    """
    t = PUNCT.sub(" ", (text or "").strip().lower()).strip()
    if mode == "first":
        return t.split(" ")[0] if t else ""
    return t


# Plusieurs LECTURES d'une même réponse. Mesuré : choisir la lecture par sonde fait passer l'écart
# moyen du top 5 de +0,260 à +0,340, sans une requête de plus. Et la meilleure lecture n'est pas
# le contenu : la CLASSE DE LONGUEUR gagne pour 18 sondes sur 49 — un LoRA change la verbosité
# avant de changer la réponse. Normaliser plus agressivement (extraire le oui/non ou le nombre)
# est en revanche CONTRE-productif : testé, l'écart du top 5 retombe à +0,203.
NORMS = {
    "key": lambda t: normalize(t, "key"),                 # contenu, casse et ponctuation écrasées
    "first": lambda t: normalize(t, "first"),             # premier mot seulement
    "trois_mots": lambda t: " ".join(normalize(t, "key").split(" ")[:3]),
    "longueur": lambda t: str(min(len(normalize(t, "key")) // 12, 12)),   # verbosité
}


def js_divergence(a: Counter, b: Counter) -> float:
    """Jensen-Shannon en base 2 entre deux distributions empiriques. Dans [0, 1]."""
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return float("nan")
    out = 0.0
    for k in keys:
        p, q = a.get(k, 0) / na, b.get(k, 0) / nb
        m = (p + q) / 2
        if p:
            out += 0.5 * p * math.log2(p / m)
        if q:
            out += 0.5 * q * math.log2(q / m)
    return max(0.0, min(1.0, out))


def load(path: Path, mode: str):
    """-> {(endpoint, sonde): Counter(reponses normalisees)}, et la liste des sondes."""
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    key = "probe" if "probe" in rows[0] else "prompt"
    dist = defaultdict(Counter)
    for r in rows:
        dist[(r["endpoint"], r[key])][normalize(r["response"], mode)] += 1
    probes = sorted({k[1] for k in dist})
    eps = sorted({k[0] for k in dist})
    return dist, probes, eps


def distance_matrix(dist, probes, eps, min_draws: int = 2):
    """Moyenne des JS par sonde. Une sonde sans assez de tirages des deux côtés est ignorée."""
    D = np.zeros((len(eps), len(eps)))
    used = np.zeros((len(eps), len(eps)), dtype=int)
    for i, j in combinations(range(len(eps)), 2):
        vals = []
        for p in probes:
            a, b = dist.get((eps[i], p)), dist.get((eps[j], p))
            if not a or not b or sum(a.values()) < min_draws or sum(b.values()) < min_draws:
                continue
            vals.append(js_divergence(a, b))
        if vals:
            D[i, j] = D[j, i] = float(np.mean(vals))
            used[i, j] = used[j, i] = len(vals)
    return D, used


def shuffle_floor(dist, probes, eps, n_iter: int = 200, seed: int = 0) -> float:
    """Plancher de bruit : JS moyenne entre deux moitiés aléatoires des tirages d'un MÊME endpoint.

    Toute distance en dessous est indiscernable de « même loi, peu d'échantillons ».
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_iter):
        e = eps[rng.integers(len(eps))]
        p = probes[rng.integers(len(probes))]
        c = dist.get((e, p))
        if not c or sum(c.values()) < 4:
            continue
        items = [k for k, n in c.items() for _ in range(n)]
        rng.shuffle(items)
        h = len(items) // 2
        vals.append(js_divergence(Counter(items[:h]), Counter(items[h:])))
    return float(np.mean(vals)) if vals else float("nan")


def probe_power(dist, probes, eps, truth):
    """Pour chaque sonde : JS moyenne sur les paires « même modèle » vs « modèles différents ».

    L'écart entre les deux est le pouvoir discriminant brut de la sonde.
    """
    out = []
    for p in probes:
        same, diff = [], []
        for a, b in combinations(eps, 2):
            ca, cb = dist.get((a, p)), dist.get((b, p))
            if not ca or not cb:
                continue
            v = js_divergence(ca, cb)
            (same if truth[a][1] == truth[b][1] else diff).append(v)
        if same and diff:
            out.append((p, float(np.mean(same)), float(np.mean(diff)),
                        float(np.mean(diff) - np.mean(same))))
    return sorted(out, key=lambda r: -r[3])


def read_truth(path: Path):
    rows = [l.strip().split(",") for l in path.open(encoding="utf-8-sig") if l.strip()]
    return {r[0]: (r[1], r[2]) for r in rows[1:] if r[0].startswith("ref-")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="reference_answers.jsonl")
    ap.add_argument("--truth", default="practice_answers.csv")
    ap.add_argument("--mode", default="key", choices=["key", "first"])
    ap.add_argument("--per-probe", action="store_true")
    a = ap.parse_args()

    dist, probes, eps = load(Path(a.answers), a.mode)
    truth = read_truth(Path(a.truth))
    print(f"{len(eps)} endpoints, {len(probes)} sondes, normalisation '{a.mode}'")

    floor = shuffle_floor(dist, probes, eps)
    print(f"plancher de bruit (mêmes tirages coupés en deux) : {floor:.4f}")

    D, used = distance_matrix(dist, probes, eps)
    lab = {e: truth.get(e, ("?", "?"))[1] for e in eps}
    print("\nmatrice de distance (JS moyenne par sonde)")
    print("            " + " ".join(f"{e[-2:]:>6s}" for e in eps))
    for i, e in enumerate(eps):
        print(f"{e} {lab[e]:3s} " + " ".join(f"{D[i, j]:6.3f}" for j in range(len(eps))))

    same = [(x, y) for x, y in combinations(eps, 2) if truth[x][1] == truth[y][1]]
    diff = [(x, y) for x, y in combinations(eps, 2) if truth[x][1] != truth[y][1]]
    g = lambda pr: D[eps.index(pr[0]), eps.index(pr[1])]  # noqa: E731
    smax, dmin = max(map(g, same)), min(map(g, diff))
    print(f"\nMÊME modèle   ({len(same):2d} paires) moy {np.mean(list(map(g, same))):.4f}  max {smax:.4f}")
    print(f"MODÈLES DIFF  ({len(diff):2d} paires) moy {np.mean(list(map(g, diff))):.4f}  min {dmin:.4f}")
    if smax < dmin:
        print(f"=> SÉPARABLE : tout seuil dans ]{smax:.4f}, {dmin:.4f}[ classe parfaitement")
    else:
        bad_s = [(f"{x}-{y}", round(g((x, y)), 3)) for x, y in same if g((x, y)) >= dmin]
        bad_d = [(f"{x}-{y}", round(g((x, y)), 3)) for x, y in diff if g((x, y)) <= smax]
        print("=> recouvrement.")
        print(f"   paires MÊME trop loin  : {bad_s}")
        print(f"   paires DIFF trop proche: {bad_d}")

    if a.per_probe:
        print("\npouvoir discriminant par sonde (écart diff - même, décroissant)")
        for p, s, d, gap in probe_power(dist, probes, eps, truth):
            print(f"  {gap:+.3f}  même {s:.3f}  diff {d:.3f}   {str(p)[:72]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
