"""Client minimal du serveur de la tâche 2 : clé lue dans .env (racine du dépôt), jamais de réessai sur 502."""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[3]
BASE = "http://34.147.229.55/33-model-equality/v1"

def key():
    if os.environ.get("API_KEY"): return os.environ["API_KEY"].strip()
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "API_KEY": return v.strip()
    raise SystemExit("pas de clé")

def session():
    s = requests.Session(); s.headers["X-API-Key"] = key(); return s

def budgets(s=None):
    s = s or session()
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in s.get(f"{BASE}/models", timeout=60).json()["data"]}

def ask(s, ep, prompt, max_tokens=256, timeout=240):
    """Retourne (status, texte, secondes, finish_reason, usage). Un seul essai sauf 429 'occupé'."""
    while True:
        t = time.time()
        try:
            r = s.post(f"{BASE}/chat/completions", timeout=timeout,
                       json={"model": ep, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]})
        except requests.RequestException as e:
            return ("EXC:" + type(e).__name__, None, time.time() - t, None, None)
        if r.status_code == 200:
            b = r.json(); c = b["choices"][0]
            return (200, c["message"]["content"], time.time() - t, c.get("finish_reason"), b.get("usage"))
        if r.status_code == 429 and ("Retry-After" in r.headers or "already has" in r.text):
            time.sleep(int(r.headers.get("Retry-After", 2))); continue
        return (r.status_code, r.text[:200], time.time() - t, None, None)

def run(jobs, out, parallel=4, paid_ok=False):
    """jobs = [(ep, probe_id, repeat, prompt, max_tokens)], reprise par cache sur (ep, probe, repeat)."""
    out = Path(out); done = set()
    if out.exists():
        for l in out.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l); done.add((r["endpoint"], r["probe"], r["repeat"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    if any(j[0].startswith("ep-") for j in jobs) and not paid_ok:
        raise SystemExit("requêtes PAYANTES demandées sans paid_ok=True")
    s = session(); lock = threading.Lock(); n = [0]; lost = []; t0 = time.time()
    fh = out.open("a", encoding="utf-8")
    def one(j):
        ep, pid, rep, prompt, mt = j
        st, txt, dt, fr, us = ask(s, ep, prompt, mt)
        with lock:
            n[0] += 1
            if st == 200:
                fh.write(json.dumps({"endpoint": ep, "probe": pid, "repeat": rep, "prompt": prompt, "max_tokens": mt,
                                     "response": txt, "finish": fr, "usage": us, "dt": round(dt, 1)}, ensure_ascii=False) + "\n"); fh.flush()
            else:
                lost.append((ep, pid, rep, st, txt))
            if n[0] % 40 == 0 or n[0] == len(jobs):
                print(f"  {n[0]}/{len(jobs)}  {n[0]/(time.time()-t0)*60:.0f}/min  perdus {len(lost)}", flush=True)
    with ThreadPoolExecutor(parallel) as p: list(p.map(one, jobs))
    fh.close()
    if lost: print("PERDUS:", lost[:8], len(lost))
    return lost
