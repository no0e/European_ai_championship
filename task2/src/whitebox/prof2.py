"""Profil fin par endpoint : T_mle interpolé, z de surprise à T_mle, z d'arrêt (EOS), longueur.
usage: prof2.py fichier.pkl [famille] [T_ref]"""
import pickle, sys, collections
import numpy as np
EXCL = ('sys_repeat', 'sys_state')
d = pickle.load(open(sys.argv[1], 'rb')); TG = np.array(d['TG'])
fam = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
Tref = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7; jr = list(TG).index(Tref)
by = collections.defaultdict(list)
for r in d['rows']:
    if r['probe'] in EXCL or (fam and r['fam'] != fam): continue
    by[r['endpoint']].append(r)
def tmle(tot):
    j = int(tot.argmax())
    if j in (0, len(TG) - 1): return TG[j]
    x = TG[j - 1:j + 2]; c = np.polyfit(x, tot[j - 1:j + 2], 2); return float(-c[1] / (2 * c[0]))
rng = np.random.default_rng(0)
print(d['repo'], d['sys'], d['quant'], '| T_ref =', Tref)
print('ep     fam     nrep  ntok  Tmle [IC95]          z_surp@Tref  z_stop@Tref  E/obs stop   rang1  nll@Tref  H@Tref')
for e, rows in sorted(by.items()):
    tots = np.stack([r['lp'].sum(1) for r in rows]); T0 = tmle(tots.sum(0))
    bs = [tmle(tots[rng.integers(0, len(rows), len(rows))].sum(0)) for _ in range(300)]
    lp = np.concatenate([r['lp'][jr] for r in rows]); H = np.concatenate([r['H'][jr] for r in rows]); V = np.concatenate([r['V'][jr] for r in rows])
    z = (-lp - H).sum() / np.sqrt(V.sum())
    pe = np.concatenate([np.exp(r['lpeos'][jr]) for r in rows]); obs = sum(r['stopped'] for r in rows)
    zs = (obs - pe.sum()) / np.sqrt((pe * (1 - pe)).sum())
    rk = np.concatenate([r['rank'] for r in rows])
    print(f"{e} {rows[0]['fam']:7s} {len(rows):4d} {len(lp):6d} {T0:5.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  {z:8.1f}  {zs:8.1f}   {pe.sum():6.1f}/{obs:3d}   {np.mean(rk==0):.3f}  {-lp.mean():.4f}  {H.mean():.4f}")
