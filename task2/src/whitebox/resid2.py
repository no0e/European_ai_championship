"""Résidus de fréquence de tokens par endpoint, E interpolé à T_mle, mode commun retiré (médiane de la famille).
usage: resid2.py fichier_watch.pkl [famille] [minE=3]"""
import pickle, sys, collections
import numpy as np
EXCL = ('sys_repeat', 'sys_state')
d = pickle.load(open(sys.argv[1], 'rb')); fam = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
minE = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
watch = d['watch']; WTS = np.array(d['WTS']); TG = np.array(d['TG']); pos = {t: i for i, t in enumerate(watch)}
by = collections.defaultdict(list)
for r in d['rows']:
    if r['probe'] in EXCL or (fam and r['fam'] != fam): continue
    by[r['endpoint']].append(r)
def tmle(tot):
    j = int(tot.argmax())
    if j in (0, len(TG) - 1): return TG[j]
    c = np.polyfit(TG[j - 1:j + 2], tot[j - 1:j + 2], 2); return float(-c[1] / (2 * c[0]))
eps = sorted(by); R = {}; Tm = {}
for e in eps:
    rows = by[e]; T = min(max(tmle(np.stack([r['lp'].sum(1) for r in rows]).sum(0)), WTS[0]), WTS[-1]); Tm[e] = T
    j = min(int(np.searchsorted(WTS, T, 'right')) - 1, len(WTS) - 2); w = (T - WTS[j]) / (WTS[j + 1] - WTS[j])
    E = np.sum([(1 - w) * r['wE'][j] + w * r['wE'][j + 1] for r in rows], 0); O = np.zeros(len(watch))
    for r in rows:
        for t in r['ids']:
            if t in pos: O[pos[t]] += 1
    R[e] = (O, E)
Etot = np.mean([R[e][1] for e in eps], 0); keep = Etot >= minE
Z = np.stack([((R[e][0] - R[e][1]) / np.sqrt(np.maximum(R[e][1], 0.5)))[keep] for e in eps])
Zc = Z - np.median(Z, 0)
print(d['repo'], '| tokens retenus :', int(keep.sum()), '| bruit attendu sur une corrélation ≈', round(100 / np.sqrt(keep.sum())))
print('ep       Tmle  chi2/ddl  chi2/ddl centré')
for i, e in enumerate(eps): print(f'{e}  {Tm[e]:.3f}  {np.mean(Z[i]**2):6.2f}  {np.mean(Zc[i]**2):6.2f}')
C = np.corrcoef(Zc)
print('corrélation des résidus centrés (x100) :')
print('       ' + ' '.join(e[-2:].rjust(4) for e in eps))
for i, a in enumerate(eps): print(a + '  ' + ' '.join(f'{100*C[i,j]:4.0f}' if i != j else '   .' for j in range(len(eps))))
