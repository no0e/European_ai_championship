"""Chargement unifié des réponses déjà collectées (aucune requête). Renvoie probe -> endpoint -> [réponses]."""
import collections, json
from pathlib import Path
T = Path(__file__).resolve().parent.parent
KIND = {'ref-01': 'plain', 'ref-02': 'fp8', 'ref-03': 'sysprompt', 'ref-04': 'lora0.7', 'ref-05': 'persona',
        'ref-06': 'watermark', 'ref-07': 'lora1.5', 'ref-08': 'temp1.0', 'ref-09': 'lora0.3'}
TRUTH = {'ref-01': 'S', 'ref-02': 'T1', 'ref-03': 'S', 'ref-04': 'T3', 'ref-05': 'S', 'ref-06': 'S', 'ref-07': 'T4',
         'ref-08': 'S', 'ref-09': 'T2'}
def _rows(fn): return [json.loads(l) for l in open(T / fn, encoding='utf-8') if l.strip()]
def load_ep():
    D = collections.defaultdict(lambda: collections.defaultdict(list)); P = {}
    for r in _rows('snapshot_answers_ep.jsonl'):
        D[r['probe']][r['endpoint']].append(r['response']); P[r['probe']] = (r['prompt'], r['max_tokens'])
    return D, P
def load_ref(P_ep):
    """Réponses ref-* ramenées au format de la collecte payante : mêmes sondes, même troncature (max_tokens x 4 caractères)."""
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    by_prompt = {v[0]: (k, v[1]) for k, v in P_ep.items()}
    for r in _rows('snapshot_answers_ref2.jsonl'):
        if r['prompt'] in by_prompt:
            D[by_prompt[r['prompt']][0]][r['endpoint']].append(r['response'])
    for r in _rows('reference_answers.jsonl'):
        if r['prompt'] in by_prompt:
            pid, mt = by_prompt[r['prompt']]
            D[pid][r['endpoint']].append(r['response'][:mt * 4])
    return D
def families(): return json.load(open(T / 'ep_families2.json'))
