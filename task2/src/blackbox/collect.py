#!/usr/bin/env python3
"""Collecte les réponses des endpoints, avec cache et `max_tokens` par sonde.

Pourquoi ne pas réutiliser `query_reference.py` tel quel : il impose `MAX_TOKENS = 256` à toutes
les sondes. Or le budget compte des RÉPONSES, pas des tokens : demander 6 tokens pour « réponds
par un seul mot » coûte exactement pareil et va ~12× plus vite. Sur 39 sondes, 782 tokens par
répétition au lieu de 9 984.

    python collect.py --endpoints ref --repeats 4            # dépistage (gratuit, illimité)
    python collect.py --endpoints ref --repeats 24 --only tie_coin,chain_mul3
    python collect.py --endpoints ep  --repeats 2 --dry-run  # TOUJOURS vérifier le coût avant

Le cache (`--out`) rend la collecte reprenable : une (endpoint, sonde, répétition) déjà présente
n'est jamais redemandée. Ne jamais supprimer ce fichier.

⚠️ La limite de 4 requêtes simultanées est PAR ÉQUIPE, toutes tâches confondues : deux collectes
lancées en parallèle se gênent et récoltent des HTTP 429.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src  # noqa: E402,F401  (KMP + stdout UTF-8 sous Windows)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probes import PROBES  # noqa: E402

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"
PARALLEL = 4                      # plafond imposé par le serveur, par équipe
KEY_FILE = Path(__file__).resolve().parents[2] / ".api_key"


def api_key() -> str:
    """La clé n'est jamais écrite dans un fichier versionné (cf. docs/22)."""
    if os.environ.get("MET_API_KEY"):
        return os.environ["MET_API_KEY"].strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    sys.exit(f"clé API introuvable : {KEY_FILE} ou la variable MET_API_KEY")


def budgets(session):
    r = session.get(f"{BASE_URL}/models", timeout=60)
    r.raise_for_status()
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in r.json()["data"]}


def load_cache(path: Path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["endpoint"], r["probe"], r["repeat"])
                for r in (json.loads(line) for line in f if line.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", default="ref", choices=["ref", "ep"],
                    help="ref = les 9 gratuits et illimités ; ep = les 33 notés (budget 100)")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--only", default=None, help="ids de sondes, séparés par des virgules")
    ap.add_argument("--only-file", default=None,
                    help="fichier avec un id de sonde PAR LIGNE. À préférer à --only : certains "
                         "identifiants sont des textes de prompt qui CONTIENNENT des virgules, "
                         "et --only les coupe en morceaux.")
    ap.add_argument("--mech", default=None, help="mécanismes, séparés par des virgules")
    ap.add_argument("--only-endpoints", default=None,
                    help="restreint aux endpoints listés (ex. ref-01,ref-02,ref-09). Pour un "
                         "criblage, juger « cette sonde sépare-t-elle fp8 ? » ne demande pas les "
                         "9 endpoints : 5 suffisent et ça coûte 44 %% de requêtes en moins.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tokens-cap", type=int, default=48,
                    help="plafonne max_tokens. Mesuré le 21/11 : la latence croît linéairement "
                         "avec max_tokens (9,4 s à 8 tokens, 13,2 s à 64, 23,6 s à 256) et le "
                         "serveur coupe vers 15 s sous charge, en FACTURANT la requête. 48 garde "
                         "une marge ; le défaut 256 du template officiel échoue systématiquement.")
    a = ap.parse_args()

    probes = list(PROBES)
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        probes = [p for p in probes if p[0] in keep]
    if a.only_file:
        keep = {line.rstrip() for line in
                Path(a.only_file).read_text(encoding="utf-8").splitlines() if line.strip()}
        missing = keep - {p[0] for p in probes}
        if missing:
            print(f"[!] {len(missing)} id(s) de {a.only_file} introuvables dans probes.py :")
            for m in sorted(missing):
                print(f"    {m[:70]}")
        probes = [p for p in probes if p[0] in keep]
    if a.mech:
        keep = {s.strip().upper() for s in a.mech.split(",")}
        probes = [p for p in probes if p[1] in keep]
    if not probes:
        sys.exit("aucune sonde sélectionnée")

    out = Path(a.out or f"answers_{a.endpoints}.jsonl")
    session = requests.Session()
    session.headers["X-API-Key"] = api_key()
    left = budgets(session)
    eps = sorted(e for e in left if e.startswith(f"{a.endpoints}-"))
    if a.only_endpoints:
        keep = {x.strip() for x in a.only_endpoints.split(",")}
        unknown = keep - set(eps)
        if unknown:
            sys.exit(f"endpoints inconnus : {sorted(unknown)}")
        eps = [e for e in eps if e in keep]
    done = load_cache(out)

    jobs, planned, skipped = [], {e: 0 for e in eps}, 0
    for rep in range(a.repeats):                       # tour par tour : un arrêt reste équilibré
        for pid, mech, prompt, mt in probes:
            for e in eps:
                if (e, pid, rep) in done:
                    continue
                if planned[e] < left[e]:
                    planned[e] += 1
                    jobs.append((e, pid, rep, prompt, min(mt, a.max_tokens_cap)))
                else:
                    skipped += 1

    cost = max(planned.values()) if planned else 0
    print(f"{len(probes)} sondes x {len(eps)} endpoints x {a.repeats} repeats")
    print(f"  déjà en cache : {len(done)} | à demander : {len(jobs)}"
          + (f" | {skipped} hors budget" if skipped else ""))
    print(f"  coût : {cost} réponses par endpoint (budget restant min "
          f"{min(left[e] for e in eps) if eps else 0})")
    print(f"  tokens demandés : {sum(j[4] for j in jobs)} (plafond serveur : 256/réponse)")
    if a.dry_run:
        print("--dry-run : rien n'est envoyé")
        return 0
    if not jobs:
        print("rien à faire")
        return 0

    lock, exhausted, n, lost = threading.Lock(), set(), [0], []
    t0 = time.time()
    fh = out.open("a", encoding="utf-8")

    def one(job):
        e, pid, rep, prompt, mt = job
        while e not in exhausted:
            try:
                r = session.post(f"{BASE_URL}/chat/completions", timeout=180,
                                 json={"model": e, "max_tokens": mt,
                                       "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException as exc:
                print(f"serveur injoignable ({type(exc).__name__}) ; nouvelle tentative dans 30 s")
                time.sleep(30)
                continue
            if r.status_code == 200:
                body = r.json()
                rec = {"endpoint": e, "probe": pid, "repeat": rep, "prompt": prompt,
                       "max_tokens": mt, "response": body["choices"][0]["message"]["content"]}
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    n[0] += 1
                    if n[0] % 50 == 0 or n[0] == len(jobs):
                        rate = n[0] / max(time.time() - t0, 1e-9)
                        eta = (len(jobs) - n[0]) / max(rate, 1e-9) / 60
                        print(f"  {n[0]}/{len(jobs)}  {rate*60:.0f}/min  reste {eta:.0f} min",
                              flush=True)
                return
            detail = r.text[:200]
            if r.status_code == 429 and "Retry-After" in r.headers:      # 4 requêtes déjà ouvertes
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429 and "already has" in detail:        # créneaux occupés
                time.sleep(2)
            elif r.status_code == 429:                                     # budget épuisé
                with lock:
                    if e not in exhausted:
                        exhausted.add(e)
                        print(f"budget épuisé sur {e} : {detail}")
            elif r.status_code in (502, 503, 504):
                # NE JAMAIS REESSAYER : le serveur facture la requete meme quand il renvoie 502.
                # Signale par l'equipe Murmell le 21/11 et verifie chez nous : 8 requetes perdues
                # sur 114, facturees sans reponse. Le template officiel reessaie toutes les 15 s en
                # boucle, ce qui permet de bruler les 100 requetes d'un endpoint sans obtenir une
                # seule reponse. On enregistre la perte et on passe a la suite.
                with lock:
                    lost.append((e, pid, rep))
                return
            else:
                print(f"{e} {pid}: HTTP {r.status_code}: {detail}", file=sys.stderr)
                return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, jobs))
    finally:
        fh.close()
    print(f"terminé en {(time.time()-t0)/60:.1f} min -> {out}")
    if lost:
        per_ep = Counter(e for e, _, _ in lost)
        print(f"[!] {len(lost)} requêtes FACTURÉES SANS RÉPONSE (502/503/504), "
              f"sur {len(per_ep)} endpoints")
        print(f"    pire endpoint : {per_ep.most_common(1)[0][0]} ({per_ep.most_common(1)[0][1]})")
        print("    relancer la même commande les redemandera (elles ne sont pas dans le cache).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
