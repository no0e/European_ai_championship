"""Soumet un CSV au serveur (une soumission / 30 min) et l'archive dans best/ avec son score.
    python k/submit.py fichier.csv "note"
"""
import csv, json, sys, time
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import api, archive

def check(path):
    rows = list(csv.DictReader(open(path)))
    eps = {r['endpoint'] for r in rows}
    assert eps == {f'ep-{i:02d}' for i in range(1, 34)}, 'il faut exactement ep-01..ep-33'
    m2f = {}
    for r in rows:
        assert m2f.setdefault(r['model'], r['family']) == r['family'], f"modèle {r['model']} dans deux familles"
    return rows

def submit(path, note=''):
    check(path)
    with open(path, 'rb') as f:
        r = requests.post('http://34.147.229.55/submit/33-model-equality', headers={'X-API-Key': api.key()},
                          files={'file': (Path(path).name, f)}, timeout=(600, 600))
    try: body = r.json()
    except ValueError: body = {'raw': r.text[:300]}
    score = body.get('score') if r.status_code == 200 and body.get('status') == 'success' else None
    e = archive.archiver(Path(path), score, note + ('' if score is not None else f' | ECHEC HTTP {r.status_code}: {str(body)[:200]}'))
    print(time.strftime('%H:%M:%S'), 'HTTP', r.status_code, json.dumps(body, ensure_ascii=False)[:400])
    print('archivé :', e['fichier'], '| score', score)
    return score, body

if __name__ == '__main__':
    submit(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
