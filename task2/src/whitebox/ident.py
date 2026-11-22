"""Résumé par famille/endpoint d'un ou plusieurs pickles de score.py : NLL par token, rang 1, T_mle."""
import pickle, sys, collections
import numpy as np
EXCL = ('sys_repeat', 'sys_state')
def summ(rows):
    lp = np.concatenate([r['lp'] for r in rows], 1); rk = np.concatenate([r['rank'] for r in rows])
    return lp.shape[1], -lp.mean(1), (rk == 0).mean(), (rk < 20).mean()
for f in sys.argv[1:]:
    d = pickle.load(open(f, 'rb')); TG = d['TG']
    by = collections.defaultdict(list)
    for r in d['rows']:
        if r['probe'] in EXCL: continue
        by[r['fam']].append(r)
    for F, rows in sorted(by.items()):
        n, nll, r1, r20 = summ(rows); j = int(nll.argmin())
        print(f"{f.split('/')[-1]:55s} {F:7s} n_tok={n:6d} nll@T1={nll[TG.index(1.0)]:.4f} min_nll={nll[j]:.4f}@T={TG[j]} rang1={r1:.4f} top20={r20:.4f} retok_ok={np.mean([r['retok_ok'] for r in rows]):.3f}")
