"""LLR par endpoint entre un modèle alternatif et la base : somme des [log p_alt(y) - log p_base(y)] à la température T.
usage: llr.py base.pkl alt.pkl [T=0.7] [famille]"""
import pickle, sys, collections
import numpy as np
EXCL = ('sys_repeat', 'sys_state')
B = pickle.load(open(sys.argv[1], 'rb')); A = pickle.load(open(sys.argv[2], 'rb'))
T = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7; fam = sys.argv[4] if len(sys.argv) > 4 else None
j = B['TG'].index(T)
alt = {r['src']: r for r in A['rows']}
by = collections.defaultdict(list)
for r in B['rows']:
    if r['probe'] in EXCL or r['src'] not in alt or (fam and r['fam'] != fam): continue
    q = alt[r['src']]
    by[r['endpoint']].append((float((q['lp'][j] - r['lp'][j]).sum()), len(r['ids']), float((q['lp'][5] - r['lp'][5]).sum())))
rng = np.random.default_rng(0)
print('alt', A['quant'], A.get('info'), '| T =', T)
print('ep      nrep   ntok    LLR   LLR/1k tok   IC95 (bootstrap réponses)    LLR@T1')
for e, L in sorted(by.items()):
    v = np.array([x[0] for x in L]); n = np.array([x[1] for x in L]); v1 = np.array([x[2] for x in L])
    idx = rng.integers(0, len(v), (2000, len(v))); b = v[idx].sum(1)
    print(f'{e} {len(v):5d} {n.sum():6d} {v.sum():8.1f} {1000*v.sum()/n.sum():8.2f}   [{np.percentile(b,2.5):8.1f}, {np.percentile(b,97.5):8.1f}]  {v1.sum():8.1f}')
