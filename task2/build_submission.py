#!/usr/bin/env python3
"""Des réponses collectées au CSV de soumission : distance, regroupement à deux niveaux, labels.

    python build_submission.py --answers answers_ref.jsonl --endpoints ref --score
    python build_submission.py --answers answers_ep.jsonl  --endpoints ep --out submission.csv

LA RÈGLE DE FORMAT, imposée ici par construction plutôt que vérifiée après coup :

    « A model label can only belong to one family. »

Le label de modèle est TOUJOURS le label de sa famille suivi d'un numéro (`A1`, `A2`, `B1`). Un
modèle ne peut donc pas appartenir à deux familles — c'est structurellement impossible, pas
simplement contrôlé. Et deux endpoints « même modèle » partagent forcément la famille, puisque le
regroupement par modèle se fait À L'INTÉRIEUR de chaque famille.

Deux niveaux, deux seuils :
    seuil LARGE  -> familles  (même modèle de base)
    seuil SERRÉ  -> modèles dans chaque famille (même checkpoint + quantification + LoRA)
Le seuil serré est forcément inférieur au large : le script le force si ce n'est pas le cas.

Les deux seuils se calibrent sur les `ref-*`, où la vérité est connue — mais attention, les 9
`ref-*` sont TOUS de la même famille : ils ne calibrent donc que le seuil SERRÉ. Le seuil large
doit être choisi sur la structure des 33 endpoints eux-mêmes (`--family-quantile`).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src  # noqa: E402,F401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from analyze import NORMS, js_divergence, read_truth  # noqa: E402


def load_raw(paths):
    raw = defaultdict(list)
    for p in paths:
        f = Path(p)
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                raw[(r["endpoint"], r["prompt"])].append(r["response"])
    return raw


def feature_distances(raw, eps, features):
    """features : [(prompt, lecture)]. -> {(a,b): distance moyenne}, et le nb de sondes partagées."""
    D, shared = {}, {}
    for a, b in combinations(eps, 2):
        vals = []
        for prompt, nm in features:
            va, vb = raw.get((a, prompt)), raw.get((b, prompt))
            if not va or not vb:
                continue
            fn = NORMS[nm]
            vals.append(js_divergence(Counter(fn(t) for t in va), Counter(fn(t) for t in vb)))
        if vals:
            D[(a, b)] = D[(b, a)] = float(np.mean(vals))
        shared[(a, b)] = shared[(b, a)] = len(vals)
    return D, shared


def components(eps, D, t):
    """Composantes connexes de (distance < t) : une partition, donc transitive par construction."""
    parent = {e: e for e in eps}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in combinations(eps, 2):
        d = D.get((a, b))
        if d is not None and d < t:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return {e: find(e) for e in eps}


def letters(i: int) -> str:
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def two_level(eps, D, t_family, t_model):
    """-> [(endpoint, famille, modèle)] avec le modèle NOMMÉ D'APRÈS sa famille.

    Le regroupement par modèle se fait à l'intérieur de chaque famille : deux endpoints de familles
    différentes ne peuvent pas recevoir le même label de modèle, même si leur distance est petite.
    """
    if t_model > t_family:
        t_model = t_family
    fam = components(eps, D, t_family)
    groups = defaultdict(list)
    for e in eps:
        groups[fam[e]].append(e)
    # familles nommées par taille décroissante, pour une sortie stable et lisible
    order = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    rows = []
    for fi, members in enumerate(order):
        flab = letters(fi)
        sub = {k: v for k, v in D.items() if k[0] in members and k[1] in members}
        mod = components(members, sub, t_model)
        mgroups = defaultdict(list)
        for e in members:
            mgroups[mod[e]].append(e)
        for mi, mm in enumerate(sorted(mgroups.values(), key=lambda g: (-len(g), g[0]))):
            for e in mm:
                rows.append((e, flab, f"{flab}{mi + 1}"))
    return sorted(rows, key=lambda r: r[0])


def two_level_declared(eps, D, declared, t_model, par_famille=None):
    """Familles IMPOSÉES (ce que le modèle déclare), modèles regroupés DANS chaque famille.

    Deux endpoints de familles différentes ne peuvent jamais recevoir le même label de modèle,
    quelle que soit leur distance : la contrainte « un label de modèle n'appartient qu'à une
    famille » reste vraie par construction. Et comparer uniquement à l'intérieur d'une famille
    supprime d'un coup les fusions absurdes entre éditeurs différents.
    """
    groups = defaultdict(list)
    for e in eps:
        groups[declared.get(e, "INCONNU")].append(e)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rows = []
    for fi, (_, members) in enumerate(order):
        flab = letters(fi)
        sub = {k: v for k, v in D.items() if k[0] in members and k[1] in members}
        t = t_model
        if par_famille is not None:
            # Un seuil global ne marche pas : chaque famille a sa propre échelle de distances, et
            # une famille entière s'effondre alors en un seul modèle pendant qu'une autre éclate.
            # Ce qui se transporte d'une famille à l'autre, c'est la PART des paires « même
            # modèle » : le jeu d'entraînement en compte 10 sur 36, soit 28 %, et chaque famille
            # des 33 contient le même jeu de variations (11 endpoints par modèle de base).
            dv = sorted(v for k, v in sub.items() if k[0] < k[1])
            if dv:
                t = float(np.quantile(dv, par_famille))
        mod = components(members, sub, t)
        mg = defaultdict(list)
        for e in members:
            mg[mod[e]].append(e)
        for mi, mm in enumerate(sorted(mg.values(), key=lambda g: (-len(g), g[0]))):
            for e in mm:
                rows.append((e, flab, f"{flab}{mi + 1}"))
    return sorted(rows, key=lambda r: r[0])


def check(rows):
    """Contrôles de format. Renvoie la liste des problèmes (vide si tout va bien)."""
    problems = []
    fam_of_model = {}
    for e, f, m in rows:
        if "," in f or "," in m:
            problems.append(f"{e} : virgule dans un label")
        if m in fam_of_model and fam_of_model[m] != f:
            problems.append(f"le modèle {m} apparaît dans les familles {fam_of_model[m]} et {f}")
        fam_of_model[m] = f
    if len({e for e, _, _ in rows}) != len(rows):
        problems.append("endpoint en double")
    return problems


def mcc(rows, truth):
    lab = {e: (f, m) for e, f, m in rows}
    eps = [e for e in lab if e in truth]
    tp = fp = fn = tn = 0
    for a, b in combinations(sorted(eps), 2):
        t = truth[a][1] == truth[b][1]
        g = lab[a] == lab[b]
        tp += t and g
        fp += (not t) and g
        fn += t and not g
        tn += (not t) and not g
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), (tp, fp, fn, tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", default=["reference_answers.jsonl", "answers_ref.jsonl"])
    ap.add_argument("--features", default="selected_features.json",
                    help="JSON [[prompt, lecture], ...]. À défaut : toutes les sondes, lecture 'key'")
    ap.add_argument("--endpoints", default="ref", choices=["ref", "ep"])
    ap.add_argument("--t-model", type=float, default=None, help="seuil serré ; sinon calibré")
    ap.add_argument("--t-family", type=float, default=None)
    ap.add_argument("--par-famille", type=float, default=None,
                    help="quantile appliqué DANS chaque famille plutôt que globalement. "
                         "Repère mesuré sur le jeu d'entraînement : 10 paires « même modèle » sur "
                         "36, soit 0,28. À préférer dès qu'il y a plusieurs familles.")
    ap.add_argument("--model-quantile", type=float, default=0.25,
                    help="à défaut de --t-model : quantile des distances observées. PRÉFÉRER ceci "
                         "à un seuil absolu. Le seuil absolu ne se transporte pas : sur les 9 "
                         "endpoints d'entraînement les paires 'même modèle' sont à 0,28-0,52, "
                         "alors que sur les 33 endpoints notés la médiane de TOUTES les paires est "
                         "à 0,29 — les modèles servis y sont plus déterministes, donc toutes les "
                         "divergences se tassent. Le quantile, lui, se transporte : calibré sur "
                         "l'entraînement il vaut 0,45-0,58 (plateau de MCC 0,524).")
    ap.add_argument("--family-quantile", type=float, default=0.75,
                    help="à défaut de seuil large : quantile des distances observées")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", action="store_true", help="scorer contre practice_answers.csv")
    ap.add_argument("--families", default=None,
                    help="JSON {endpoint: famille} issu de families.py. Les familles viennent "
                         "alors de ce que le modèle DÉCLARE être, pas d'un seuil de distance — "
                         "la composante famille vaut 10 %% du score et tombe à zéro si on met tout "
                         "le monde ensemble. Un endpoint absent ou INCONNU est rattaché à la "
                         "famille de son plus proche voisin.")
    a = ap.parse_args()

    raw = load_raw(a.answers)
    eps = sorted({e for e, _ in raw if e.startswith(f"{a.endpoints}-")})
    if not eps:
        sys.exit(f"aucun endpoint '{a.endpoints}-' dans {a.answers}")

    if Path(a.features).exists():
        features = [tuple(x) for x in json.loads(Path(a.features).read_text(encoding="utf-8"))]
    else:
        features = [(p, "key") for p in {p for _, p in raw}]
        print(f"[!] {a.features} absent : toutes les sondes en lecture 'key' ({len(features)})")

    D, shared = feature_distances(raw, eps, features)

    declared = None
    if a.families and Path(a.families).exists():
        declared = json.loads(Path(a.families).read_text(encoding="utf-8"))
        # rattacher les non identifiés au plus proche voisin DÉJÀ identifié
        known = {e: f for e, f in declared.items() if e in eps and f != "INCONNU"}
        for e in eps:
            if e in known:
                continue
            cands = [(D.get((e, o), 9), o) for o in known]
            if cands:
                d, o = min(cands)
                known[e] = known[o]
                print(f"  {e} non identifié -> famille de {o} ({known[o]}, distance {d:.3f})")
        declared = known
    ns = [v for k, v in shared.items() if k[0] < k[1]]
    print(f"{len(eps)} endpoints, {len(features)} caractéristiques, "
          f"sondes partagées par paire : min {min(ns)} méd {int(np.median(ns))}")
    if min(ns) == 0:
        print("[!] des paires n'ont AUCUNE sonde en commun : elles sont incomparables.")

    vals = sorted(D[(x, y)] for x, y in combinations(eps, 2))
    t_fam = a.t_family if a.t_family is not None else float(np.quantile(vals, a.family_quantile))
    t_mod = a.t_model if a.t_model is not None else float(np.quantile(vals, a.model_quantile))
    if declared:
        rows = two_level_declared(eps, D, declared, t_mod, a.par_famille)
    else:
        rows = two_level(eps, D, t_fam, t_mod)
    print(f"seuils : famille {t_fam:.4f}, modèle {t_mod:.4f}")
    print(f"-> {len({f for _, f, _ in rows})} familles, {len({m for _, _, m in rows})} modèles")

    problems = check(rows)
    print("format :", "OK" if not problems else f"{len(problems)} PROBLÈME(S)")
    for p in problems:
        print("   ", p)

    if a.score:
        truth = read_truth(Path("practice_answers.csv"))
        if truth:
            s, (tp, fp, fn, tn) = mcc(rows, truth)
            print(f"MCC sur les paires de référence : {s:.3f}  "
                  f"(justes {tp + tn}, faux {fp + fn} : {fp} à tort 'même', {fn} à tort 'différent')")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            fh.write("endpoint,family,model\n")
            for e, f, m in rows:
                fh.write(f"{e},{f},{m}\n")
        print(f"écrit : {a.out}")
    else:
        for e, f, m in rows[:12]:
            print(f"  {e},{f},{m}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
