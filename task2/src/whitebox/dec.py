"""Teste des hypothèses de décodage : part de tokens hors de l'ensemble autorisé, par endpoint.
usage: dec.py fichier.pkl [famille]"""
import pickle, sys, collections
import numpy as np
EXCL = ('sys_repeat', 'sys_state')
d = pickle.load(open(sys.argv[1], 'rb')); fam = sys.argv[2] if len(sys.argv) > 2 else None
by = collections.defaultdict(list)
for r in d['rows']:
    if r['probe'] in EXCL or (fam and r['fam'] != fam): continue
    by[r['endpoint']].append(r)
HYP = [(1.0, 0.7, 0, 1.0), (1.0, 0.7, 20, 1.0), (1.0, 0.7, 0, 0.9), (1.0, 0.7, 0, 0.8), (1.0, 0.7, 20, 0.8), (1.1, 0.7, 20, 1.0), (1.1, 0.7, 20, 0.8), (1.1, 0.7, 0, 0.8), (1.05, 0.7, 20, 0.8),
       (1.0, 0.6, 0, 0.9), (1.0, 1.0, 0, 0.9), (1.0, 1.0, 20, 0.8), (1.1, 1.0, 20, 0.8), (1.1, 0.8, 20, 0.8), (1.0, 0.8, 0, 0.9), (1.0, 1.0, 0, 0.95), (1.0, 0.7, 0, 0.95), (1.0, 0.7, 50, 1.0), (1.0, 0.7, 40, 1.0)]
print(d['repo'], d['sys'], '| colonnes : (pénalité, T, top-k, top-p) -> % de tokens hors ensemble')
print('ep     ' + ' '.join(f'{h[0]:.2f}/{h[1]}/{h[2]}/{h[3]}'.rjust(15) for h in HYP))
for e, rows in sorted(by.items()):
    out = []
    for rp, T, k, p in HYP:
        rk = np.concatenate([r['dec'][(rp, T)]['rank'] for r in rows]); ab = np.concatenate([r['dec'][(rp, T)]['above'] for r in rows]); m20 = np.concatenate([r['dec'][(rp, T)]['m20'] for r in rows])
        bad = np.zeros(len(rk), bool)
        if k == 20: bad |= rk >= k; ab = ab / np.maximum(m20, 1e-9)
        elif k: bad |= rk >= k
        if p < 1: bad |= ab >= p
        out.append(100 * bad.mean())
    print(e + ' ' + ' '.join(f'{x:15.2f}' for x in out))
