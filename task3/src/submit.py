"""Submit files to task 3 one after another, honouring the server rate limit (waits on "Wait N").
Key: $T3_API_KEY (or API_KEY in $T3_WORK/.submit_t3.py), never printed. URL: $T3_SUBMIT_URL.
Logs: $T3_WORK/output/submit_log.txt and $T3_WORK/logs/submit.jsonl.
Usage: python src/submit.py file1.pt [file2.pt ...]"""
import json, os, re, sys, time
import requests
ROOT = os.path.abspath(os.environ.get("T3_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KEY = os.environ.get("T3_API_KEY") or re.search(r'^API_KEY\s*=\s*["\']([^"\']+)["\']', open(os.path.join(ROOT, ".submit_t3.py")).read(), re.M).group(1)
URL = os.environ.get("T3_SUBMIT_URL", "http://34.147.229.55/submit/31-unlearning")
def log(line):
    try:
        os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
        with open(os.path.join(ROOT, "output", "submit_log.txt"), "a") as f: f.write(line + "\n")
    except OSError as e: print("journal non écrit (quota ?) :", e, flush=True)
for path in sys.argv[1:]:
    path = os.path.abspath(path); rel = os.path.relpath(path, ROOT); tries = 0
    while True:
        try:
            with open(path, "rb") as f:
                r = requests.post(URL, headers={"X-API-Key": KEY}, files={"file": (os.path.basename(path), f)}, timeout=(10, 600))
            try: body = r.json()
            except ValueError: body = {"raw": r.text[:300]}
        except requests.RequestException as e:
            body = {"error": repr(e)[:200]}; r = None
        txt = json.dumps(body)
        m = re.search(r"[Ww]ait (\d+)", txt)
        if m and (r is None or r.status_code != 200):
            time.sleep(max(1, int(m.group(1)) - 1)); continue
        if "error" in body and tries < 3:      # coupure réseau à l'envoi : on réessaie (si le serveur l'avait reçu, il répondra « Wait »)
            tries += 1; time.sleep(30); continue
        stamp = time.strftime("%H:%M")
        print(stamp, rel, body.get("score"), body.get("status", body), flush=True)      # d'abord la sortie standard (dans le home)
        log(f"{stamp} {rel} | BODY {txt}")
        try:
            os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
            with open(os.path.join(ROOT, "logs", "submit.jsonl"), "a") as f: f.write(json.dumps(dict(time=stamp, file=rel, body=body)) + "\n")
        except OSError: pass
        break
