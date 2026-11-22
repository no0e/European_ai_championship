"""Envoie un fichier de soumission DÉJÀ construit (assemble.py --out ...), sans le reconstruire.

Clé : variable d'environnement WMLOC_API_KEY, sinon fichier .env (WMLOC_API_KEY=... ou API_KEY=...) dans un dossier
parent (racine du dépôt). La clé n'est jamais affichée ni écrite. Chaque envoi est consigné dans soumissions.csv
(src/submit.py::log_result).

    python submit_file.py out/sub_v3.pt --note "v3 : CV 0,7085, +0,032 apparié sur v2"
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch

from assemble import BASE_URL, TASK_ID
from common import HERE, WORK


def read_key():
    key = os.environ.get("WMLOC_API_KEY")
    if key:
        return key.strip()
    for d in [HERE, *HERE.parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() in ("WMLOC_API_KEY", "API_KEY") and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    p = Path(a.path)
    sub = torch.load(str(p), weights_only=True)                 # même contrôle que assemble.py
    assert len(set(sub["image_names"])) == 1020 and tuple(sub["masks"].shape) == (1020, 128, 128)
    assert sub["masks"].dtype == torch.bool and os.path.getsize(p) / 2 ** 20 <= 20
    sha = hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    key = read_key()
    if not key:
        sys.exit("clé absente (WMLOC_API_KEY ou .env) : rien n'est envoyé.")
    import requests
    with open(p, "rb") as f:
        r = requests.post(f"{BASE_URL}/submit/{TASK_ID}", headers={"X-API-Key": key},
                          files={"file": (p.name, f, "application/octet-stream")}, timeout=(10, 300))
    try:
        rep = r.json()
    except Exception:
        rep = r.text[:300]
    print("HTTP", r.status_code, str(rep).replace(key, "***"))          # la clé n'est jamais affichée
    score = rep.get("score", "") if isinstance(rep, dict) else ""
    import csv, time
    log = WORK / "out" / "submissions.csv"; log.parent.mkdir(parents=True, exist_ok=True); new = not log.exists()
    with open(log, "a", newline="") as fh:                       # same columns as results/submissions.csv
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "task", "score", "submission", "seed", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "t1-watermark-loc", score, f"{p.name} sha1 {sha}", "",
                    f"HTTP {r.status_code} ; {a.note}"])


if __name__ == "__main__":
    main()
