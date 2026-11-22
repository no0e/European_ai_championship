#!/usr/bin/env python3
"""Soumission structurée : familles déclarées, puis variantes cosmétiques forcées dans leur modèle.

    python solve.py --out submission.csv --quantile 0.28

Pourquoi ne pas se contenter d'un clustering par distance. Mesuré sur les 9 endpoints
d'entraînement, dont la vérité est connue : `ref-03` (prompt système, MÊME modèle que `ref-01`)
est à 0,730 de `ref-01`, tandis que `ref-02` (fp8, modèle DIFFÉRENT) n'est qu'à 0,628. L'ordre est
inversé, donc aucun seuil ne peut marcher : une distance sur les sorties mesure d'abord la couche
cosmétique, pas les poids. C'est le mécanisme qui plafonne toute méthode par seuil à MCC 0,524 —
exactement le score de « tout juste sauf fp8 et LoRA 0,3 ».

La correction consiste à traiter séparément ce que la consigne déclare non identifiant :
« watermark, prompt système, stratégie de décodage (greedy ou beam) ne changent pas le modèle ».

  1. PROMPT SYSTÈME / PERSONA — repérés non pas par l'ampleur de l'écart de style (règle testée et
     FAUSSE : sur l'entraînement les deux plus gros écarts sont `ref-07` LoRA 1.5 et `ref-04`
     LoRA 0,7, donc des modèles différents) mais par une INCOHÉRENCE D'IDENTITÉ : un endpoint qui
     se déclare d'un autre éditeur que sa famille a un prompt système qui le renomme. Mesuré :
     `ep-02` et `ep-23` dans QWEN_A, `ep-08` et `ep-20` dans QWEN_B, `ep-16` dans LLAMA —
     `ep-08` se déclare même tour à tour « created by Anthropic » et « created by Alibaba Cloud ».
     Ceux-là sont forcés dans le modèle principal de leur famille.

  2. DÉCODAGE DÉTERMINISTE (greedy, beam) — repéré par des réponses identiques quand on répète le
     même prompt. Signal mesuré sur les 33 : `ep-06` à 0,89 et `ep-32` à 0,86 de réponses répétées
     identiques, contre 0,56-0,72 pour le reste de QWEN_B. Non vérifiable sur l'entraînement, qui
     ne contient pas ces variations : traité en option (`--greedy`).

  3. le reste est regroupé par distance DANS chaque famille, à part fixée (voir `--quantile`).

Hypothèse de structure, non confirmée : les trois familles font 11 endpoints chacune et la liste
officielle des variations en compte justement onze (plain, 2 quantifications, 3 forces de LoRA,
prompt système, persona, watermark, greedy, beam). Elle prédit 6 endpoints cosmétiques et
5 variantes de poids par famille, soit 15 paires « même modèle » sur 55 = 27,3 % — à comparer aux
10 sur 36 = 27,8 % mesurés sur l'entraînement. Les deux concordent, ce qui ne les rend pas vraies.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src  # noqa: E402,F401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from build_submission import components, feature_distances, letters, load_raw  # noqa: E402

VENDEURS = {
    "QWEN_A": r"\bqwen\b|\balibaba\b|\btongyi\b",
    "QWEN_B": r"\bqwen\b|\balibaba\b|\btongyi\b",
    "LLAMA": r"\bllama\b|\bmeta\b",
}
AUTRES = r"\bclaude\b|\banthropic\b|\bchatgpt\b|\bgpt-?\d|\bopenai\b|\bgemini\b|\bmistral\b"
IDENT = ("LLM are you", "who created you", "what model are you")


def persona(rows, fam):
    """Endpoints dont l'identité déclarée contredit celle de leur famille -> prompt système.

    La consigne range le prompt système parmi les différences qui NE changent PAS le modèle : ces
    endpoints doivent donc rejoindre le modèle principal, alors qu'une distance les en éloigne.
    """
    dit = defaultdict(list)
    for r in rows:
        if any(m in r["prompt"] for m in IDENT):
            dit[r["endpoint"]].append(r["response"].lower())
    out = {}
    for e, textes in dit.items():
        f = fam.get(e)
        if f not in VENDEURS:
            continue
        txt = " ".join(textes)
        sien = bool(re.search(VENDEURS[f], txt))
        autre = bool(re.search(AUTRES, txt))
        if autre:
            out[e] = "autre editeur declare"
        elif not sien:
            out[e] = "aucune identite de sa famille"
    return out


def stabilite(rows):
    """Part des paires de tirages répétés qui sont identiques -> décodage déterministe."""
    pool = defaultdict(list)
    for r in rows:
        pool[(r["endpoint"], r["prompt"])].append(r["response"])
    out = {}
    for e in {e for e, _ in pool}:
        tot = ident = 0
        for (autre, _), v in pool.items():
            if autre != e:
                continue
            for x, y in combinations(v, 2):
                tot += 1
                ident += x.strip() == y.strip()
        out[e] = ident / tot if tot else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="answers_ep.jsonl")
    ap.add_argument("--families", default="ep_families2.json")
    ap.add_argument("--features", default="ep_features.json")
    ap.add_argument("--quantile", type=float, default=0.28,
                    help="part des paires « même modèle » attendue DANS chaque famille")
    ap.add_argument("--greedy", type=int, default=0,
                    help="nombre d'endpoints les plus stables par famille à traiter comme un "
                         "décodage déterministe, donc cosmétiques (0 = ne pas s'en servir)")
    ap.add_argument("--taille", type=int, default=None,
                    help="taille IMPOSÉE du modèle principal de chaque famille ; les autres "
                         "endpoints deviennent des singletons. Supprime tout seuil : la structure "
                         "donne directement 6 (plain + prompt système + persona + watermark + "
                         "greedy + beam), les 5 restants étant les variantes de poids. On part "
                         "des cosmétiques détectés, puis on complète par la clique la plus serrée "
                         "parmi les endpoints restants.")
    ap.add_argument("--out", default="submission_solve.csv")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.answers, encoding="utf-8") if l.strip()]
    fam = json.loads(Path(a.families).read_text(encoding="utf-8"))
    raw = load_raw([a.answers])
    eps = sorted({e for e, _ in raw if e.startswith("ep-")})
    feats = [tuple(x) for x in json.loads(Path(a.features).read_text(encoding="utf-8"))]
    D, _ = feature_distances(raw, eps, feats)

    forces = persona(rows, fam)
    stab = stabilite(rows)
    groupes = defaultdict(list)
    for e in eps:
        groupes[fam[e]].append(e)

    out_rows = []
    for fi, (f, membres) in enumerate(sorted(groupes.items(),
                                             key=lambda kv: (-len(kv[1]), kv[0]))):
        flab = letters(fi)
        cosmet = {e for e in membres if e in forces}
        if a.greedy:
            for e in sorted(membres, key=lambda x: -stab.get(x, 0))[:a.greedy]:
                cosmet.add(e)
        libres = [e for e in membres if e not in cosmet]
        sub = {k: v for k, v in D.items() if k[0] in libres and k[1] in libres}
        if a.taille:
            # Taille connue : on énumère les sous-ensembles de `libres` qui complètent les
            # cosmétiques et on garde celui dont la distance interne moyenne est la plus faible.
            # C(11,6) = 462 au pire, donc exhaustif. Aucun seuil n'intervient.
            k = max(a.taille - len(cosmet), 0)
            if k and len(libres) > k:
                _, choix = min((float(np.mean([D[(x, y)] for x, y in combinations(s, 2)]))
                                if len(s) > 1 else 0.0, s)
                               for s in combinations(sorted(libres), k))
            else:
                choix = tuple(sorted(libres))[:k]
            ordre = [sorted(choix)] + [[e] for e in sorted(libres) if e not in choix]
        else:
            dv = sorted(v for k2, v in sub.items() if k2[0] < k2[1])
            t = float(np.quantile(dv, a.quantile)) if dv else 0.0
            comp = components(libres, sub, t)
            mg = defaultdict(list)
            for e in libres:
                mg[comp[e]].append(e)
            ordre = sorted(mg.values(), key=lambda g: (-len(g), g[0]))
        # les cosmétiques rejoignent le plus gros groupe : c'est là qu'est le modèle « nu »
        if ordre:
            ordre[0] = sorted(ordre[0] + sorted(cosmet))
        elif cosmet:
            ordre = [sorted(cosmet)]
        print(f"  {flab} ({f}) : {len(ordre)} modeles  " +
              "  ".join(f"[{' '.join(g)}]" for g in ordre))
        if cosmet:
            print("      forces dans le modele principal : " +
                  ", ".join(f"{e} ({forces.get(e, 'decodage deterministe')})"
                            for e in sorted(cosmet)))
        for mi, g in enumerate(ordre):
            for e in g:
                out_rows.append((e, flab, f"{flab}{mi + 1}"))

    out_rows.sort()
    fam_of = {}
    for e, f, m in out_rows:
        assert fam_of.setdefault(m, f) == f, f"le modele {m} apparait dans deux familles"
    assert len({e for e, _, _ in out_rows}) == len(out_rows) == 33, "33 endpoints attendus"
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        fh.write("endpoint,family,model\n")
        for e, f, m in out_rows:
            fh.write(f"{e},{f},{m}\n")
    paires = sum(n * (n - 1) // 2 for n in Counter(m for _, _, m in out_rows).values())
    print(f"\n-> {len({f for _, f, _ in out_rows})} familles, "
          f"{len({m for _, _, m in out_rows})} modeles, {paires} paires « meme modele »")
    print("format : OK")
    print(f"ecrit : {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
