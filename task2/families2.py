#!/usr/bin/env python3
"""Famille de chaque endpoint, lue sur des HABITUDES du modèle de base et non sur l'éditeur cité.

Pourquoi une deuxième version. La première (`families.py`) prenait l'éditeur déclaré pour la
famille, et donnait QWEN 18 / LLAMA 11 / CLAUDE 3. Mesuré sur les réponses, c'est faux deux fois :

  - « CLAUDE » n'est pas une famille, c'est un PERSONA. `ep-08` se déclare tour à tour
    « created by Anthropic » et « created by Alibaba Cloud » ; `ep-33` cite les deux ; `ep-23` se
    présente comme « an AI-powered journalist ». Un prompt système qui renomme le modèle ne change
    pas ses poids — la consigne le dit : prompt système et persona sont le MÊME modèle.
  - les 18 « QWEN » se scindent nettement en deux sur une habitude que le persona ne touche pas :
    la réaction au préfixe d'injection. Neuf refusent (« the text you provided does not contain any
    information about who created me »), neuf répondent directement. Et le partage se recoupe sur
    un second signal indépendant : les premiers donnent une date de cutoff chiffrée (février 2023,
    septembre 2022, octobre 2021…), les seconds refusent systématiquement d'en donner.

Les 11 Llama sont le groupe le plus sûr : dix d'entre eux renvoient la même phrase au caractère
près, « My data cutoff date is December 2023, but I have access to more recent information via
internet search. » — une réponse mémorisée, donc une signature de poids, pas de style.

Ce que ça vaut. La composante famille pèse 10 % du score. Avec une seule famille elle est nulle
par construction ; avec les familles d'éditeur le leaderboard a renvoyé 0,05 au total, ce qui
situe le MCC famille vers 0,5. Le découpage ci-dessous est la correction la plus directe.

Les endpoints ambigus (refus d'injection à 0,5) sont tranchés par la distance : on les rattache
au groupe dont ils sont le plus proche, ce qui ne crée pas de famille supplémentaire.

    python families2.py                 # tableau et écriture de ep_families2.json
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src  # noqa: E402,F401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRIP = "))))]]]]}}}};*/"

# Marqueurs du refus d'injection. Ce n'est pas « le modèle refuse de répondre » au sens d'un
# garde-fou : c'est « le modèle lit le préfixe comme du texte à commenter » plutôt que comme une
# consigne à suivre. Habitude du modèle de base, pas du prompt système.
REFUS = re.compile(r"\b(gibberish|does not contain|doesn't contain|confusion|formatting issues|"
                   r"no clear|series of|curly brace|mix of symbols|incorrectly formatted|"
                   r"cannot accurately|some confusion)\b", re.I)
DATE = re.compile(r"\b(19|20)\d{2}\b")
MOIS = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|"
                  r"november|december)\b", re.I)
LLAMA = re.compile(r"\bllama\b|\bmeta ai\b|\bdeveloped (?:by|at) meta\b|\bmeta\b", re.I)


def profils(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if STRIP in r["prompt"]:
            by[r["endpoint"]]["inj"].append(r["response"])
        elif "cutoff" in r["prompt"]:
            by[r["endpoint"]]["cut"].append(r["response"])
        elif "LLM are you" in r["prompt"]:
            by[r["endpoint"]]["who"].append(r["response"])
    out = {}
    for e, d in by.items():
        inj, cut = d["inj"], d["cut"]
        out[e] = {
            "refus": sum(bool(REFUS.search(t)) for t in inj) / max(len(inj), 1),
            "date": sum(bool(DATE.search(t) or MOIS.search(t)) for t in cut) / max(len(cut), 1),
            "llama": bool(LLAMA.search(" ".join(d["who"] + inj + cut))),
        }
    return out


def familles(rows, D=None):
    """-> {endpoint: famille}, {endpoint: profil}. Trois groupes, les ambigus par distance."""
    prof = profils(rows)
    fam, ambigus = {}, []
    for e, p in prof.items():
        if p["llama"]:
            fam[e] = "LLAMA"
        elif p["refus"] >= 0.75:
            fam[e] = "QWEN_A"
        elif p["refus"] <= 0.25:
            fam[e] = "QWEN_B"
        else:
            ambigus.append(e)
    for e in ambigus:
        if D:
            # plus proche voisin parmi les endpoints deja classes, hors Llama : l'ambiguite ne
            # porte que sur le partage A/B.
            cands = [(D.get((e, o), 9), o) for o in fam if fam[o] != "LLAMA"]
            if cands:
                fam[e] = fam[min(cands)[1]]
                continue
        # a defaut de distance : la date de cutoff, second signal independant du partage
        fam[e] = "QWEN_A" if prof[e]["date"] > 0 else "QWEN_B"
    return fam, prof


def main():
    rows = [json.loads(l) for l in open("answers_ep.jsonl", encoding="utf-8") if l.strip()]
    D = None
    try:
        from build_submission import feature_distances, load_raw
        raw = load_raw(["answers_ep.jsonl"])
        eps = sorted({e for e, _ in raw if e.startswith("ep-")})
        feats = [tuple(x) for x in json.loads(
            Path("ep_features.json").read_text(encoding="utf-8"))]
        D, _ = feature_distances(raw, eps, feats)
    except Exception as exc:
        print(f"[!] distances indisponibles ({type(exc).__name__}) : les ambigus seront "
              f"tranchés par la date de cutoff")
    fam, prof = familles(rows, D)

    groups = defaultdict(list)
    for e, f in fam.items():
        groups[f].append(e)
    print(f"{len(fam)} endpoints, {len(groups)} familles\n")
    for f, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        g.sort(key=lambda x: int(x.split("-")[1]))
        print(f"  {f:8s} ({len(g):2d}) {' '.join(g)}")
    amb = [e for e in fam if 0.25 < prof[e]["refus"] < 0.75 and not prof[e]["llama"]]
    if amb:
        print(f"\n  tranchés par distance ({len(amb)}) : " +
              " ".join(f"{e}->{fam[e]}" for e in sorted(amb)))
    print("\n  second signal, date de cutoff chiffrée (doit suivre le partage A/B) :")
    for f in ("QWEN_A", "QWEN_B"):
        vals = [prof[e]["date"] for e in groups.get(f, [])]
        if vals:
            print(f"    {f} : {sum(vals) / len(vals):.2f} en moyenne")

    Path("ep_families2.json").write_text(json.dumps(fam, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("\nécrit : ep_families2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
