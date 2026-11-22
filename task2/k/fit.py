"""Quelles structures (groupe principal + isolés) sont compatibles avec TOUS les scores publics observés ?
Approximation : on calcule le score sur toutes les paires, alors que le score public n'en couvre qu'une partie."""
import csv, itertools, json, math, sys
from pathlib import Path
T = Path(__file__).resolve().parent.parent
FAM = json.load(open(T / 'ep_families2.json'))
EPS = {F: sorted(e[-2:] for e in FAM if FAM[e] == F) for F in ('QWEN_A', 'QWEN_B', 'LLAMA')}
def pred_groups(path):
    g = {}
    for r in csv.DictReader(open(path)): g.setdefault(r['model'], []).append(r['endpoint'][-2:])
    return {e: m for m, es in g.items() for e in es}
def counts(truth, pred, eps):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t = truth[a] == truth[b]; p = pred[a] == pred[b]
        tp += t and p; fp += (not t) and p; fn += t and (not p); tn += (not t) and (not p)
    return tp, fp, fn, tn
def score(c):
    tp, fp, fn, tn = c; d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * max(0.0, (tp * tn - fp * fn) / d if d else 0.0)
def truths(F, must, sizes, pairs=()):
    """Vérités candidates : un groupe principal contenant `must`, le reste isolé ; en option une paire hors groupe."""
    others = [e for e in EPS[F] if e not in must]
    for m in sizes:
        for extra in itertools.combinations(others, m - len(must)):
            M = set(must) | set(extra)
            for pr in [None] + [p for p in pairs if not (set(p) & M)]:
                t = {e: ('M' if e in M else e) for e in EPS[F]}
                if pr:
                    for e in pr: t[e] = 'P'
                yield (tuple(sorted(M)), pr), t
OBS = [l.split() for l in open(T / 'k' / 'obs.txt') if l.strip() and not l.startswith('#')]
PRED = [(pred_groups(T / p), float(s)) for p, s in OBS]
if __name__ == '__main__':
    QA = {e: ('M' if e in ('01', '02', '04', '12', '23', '33') else e) for e in EPS['QWEN_A']}
    cA = [counts(QA, p, EPS['QWEN_A']) for p, _ in PRED]
    TB = [(k, [counts(t, p, EPS['QWEN_B']) for p, _ in PRED]) for k, t in truths('QWEN_B', ['08', '20'], (5, 6, 7), pairs=[('07', '32'), ('21', '32')])]
    TL = [(k, [counts(t, p, EPS['LLAMA']) for p, _ in PRED]) for k, t in truths('LLAMA', ['16'], (4, 5, 6, 7, 8), pairs=[('05', '11'), ('10', '15'), ('09', '19')])]
    print(len(TB), 'x', len(TL), 'combinaisons')
    res = []
    for kb, cb in TB:
        for kl, cl in TL:
            err = 0.0; pr = []
            for i, (_, s) in enumerate(PRED):
                sc = score(tuple(a + b + c for a, b, c in zip(cA[i], cb[i], cl[i]))); pr.append(sc); err += abs(sc - s)
            res.append((err, kb, kl, pr))
    res.sort(key=lambda r: r[0])
    for err, kb, kl, pr in res[:int(sys.argv[1]) if len(sys.argv) > 1 else 25]:
        print(f'{err:.3f}  B={" ".join(kb[0])}{" +paire " + "-".join(kb[1]) if kb[1] else ""} | L={" ".join(kl[0])}{" +paire " + "-".join(kl[1]) if kl[1] else ""} | ' + ' '.join(f'{x:.2f}' for x in pr))
    import collections
    top = [r for r in res if r[0] <= res[0][0] + 0.05]
    print(len(top), 'combinaisons à moins de 0,05 du meilleur écart ; fréquence d\'appartenance au groupe principal :')
    for F, idx in (('QWEN_B', 1), ('LLAMA', 2)):
        c = collections.Counter(e for r in top for e in r[idx][0]); print(' ', F, {e: round(c[e] / len(top), 2) for e in EPS[F]})
