"""Aide à la soumission + journal des soumissions.

À ADAPTER le jour J au format réel du serveur CISPA (endpoint, auth, nom de fichier).
On ne connaît pas le format à l'avance -> ici on gère seulement :
  - l'écriture d'un fichier de soumission propre et versionné,
  - un journal CSV (score, seed, note) pour ne jamais perdre la meilleure soumission.

Règle d'or : vérifier le score EN LOCAL avant de soumettre ; garder une marge de
soumissions pour la fin (fenêtres parfois limitées).
"""
from __future__ import annotations
import os, csv, json, time, hashlib
import numpy as np


def save_submission(array_or_obj, out_dir="submissions", prefix="sub", meta: dict | None = None):
    """Sauvegarde une soumission horodatée + un .json de métadonnées.

    Retourne le chemin du fichier. Supporte np.ndarray (-> .npy) ou objet JSON-sérialisable.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    if isinstance(array_or_obj, np.ndarray):
        path = os.path.join(out_dir, f"{prefix}_{ts}.npy")
        np.save(path, array_or_obj)
    else:
        path = os.path.join(out_dir, f"{prefix}_{ts}.json")
        with open(path, "w") as f:
            json.dump(array_or_obj, f)
    digest = _sha1(path)
    with open(path + ".meta.json", "w") as f:
        json.dump({"ts": ts, "sha1": digest, **(meta or {})}, f, indent=2)
    print(f"[submit] écrit {path} (sha1 {digest[:10]})")
    return path


def log_result(csv_path, task, score, submission_path, seed=None, note=""):
    """Ajoute une ligne au journal des soumissions (crée l'en-tête si besoin)."""
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "task", "score", "submission", "seed", "note"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), task, score,
                    submission_path, seed if seed is not None else "", note])
    print(f"[submit] journal <- {task} score={score}")


def best_so_far(csv_path, task=None):
    """Lit le journal et renvoie la meilleure ligne (score max)."""
    if not os.path.exists(csv_path):
        return None
    best = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if task and row["task"] != task:
                continue
            try:
                sc = float(row["score"])
            except ValueError:
                continue
            if best is None or sc > float(best["score"]):
                best = row
    return best


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --- squelette d'envoi HTTP, à compléter avec le vrai endpoint ---
def post_submission(endpoint, path, token=None, field="file"):
    """Envoi générique multipart. NE PAS supposer le format : à ajuster le jour J."""
    import requests  # installé par setup_env.sh
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(path, "rb") as f:
        r = requests.post(endpoint, files={field: f}, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json() if "application/json" in r.headers.get("content-type", "") else r.text


# --- format observé à Paris 2025 (dépôt floriandgg/cispa, task_template.py) ---
# POST {base}/submit/{task_id}, en-tête X-API-Key, multipart champ "file" ; réponse JSON avec
# "submission_id" ; HTTP 413 = fichier trop gros. Requêtes au modèle cible (budget limité) :
# POST {base}/{task_id}/logits, champ "npz". À re-vérifier sur le template 2026.
def cispa_submit(base_url, task_id, path, api_key, timeout=(10, 120)):
    """Soumission au serveur CISPA (format 2025). Renvoie le JSON du serveur."""
    import requests
    with open(path, "rb") as f:
        r = requests.post(f"{base_url.rstrip('/')}/submit/{task_id}", headers={"X-API-Key": api_key},
                          files={"file": (os.path.basename(path), f)}, timeout=timeout)
    if r.status_code == 413:
        raise RuntimeError("HTTP 413 : fichier trop gros")
    try:
        body = r.json()
    except Exception:
        body = {"raw_text": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} : {body}")
    return body


def cispa_query(base_url, task_id, path, api_key, endpoint="logits", field="npz", timeout=(10, 120)):
    """Requête au modèle cible (format 2025 : /{task_id}/logits). Chaque appel consomme du budget."""
    import requests
    with open(path, "rb") as f:
        r = requests.post(f"{base_url.rstrip('/')}/{task_id}/{endpoint}", headers={"X-API-Key": api_key},
                          files={field: (os.path.basename(path), f, "application/octet-stream")},
                          timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:500]}")
    return r.json()
