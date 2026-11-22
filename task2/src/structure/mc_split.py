"""Overfitting check: Monte Carlo over the unknown public/hidden split of the pairs.
- candidate truths per family: one main group (always containing the disguises identified in white box, never the
  detected fp8) + endpoints alone;
- public splits drawn at random (a fraction p of the pairs);
- each (truth, split) is weighted by how well it reproduces every public score we observed (scores are rounded to
  0.05; soft likelihood with sigma on the distance to the rounding interval);
- output: P(endpoint in the main group) and, for each candidate partition, the expected score on the HIDDEN pairs
  (the final ranking) and the chance that its public score would exceed 0.85.
usage (from task2/): python3 src/structure/mc_split.py [p=0.3] [nS=300] [fixe|libre] [sigma=0.03]"""
import csv, itertools, json, sys
import numpy as np
from pathlib import Path
T = Path(__file__).resolve().parents[2]          # task2/
p = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
nS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
FREE_COSM = len(sys.argv) > 3 and sys.argv[3] == 'libre'
SIG = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03          # do not force watermark and T=1 into the group
FAM = json.load(open(T / 'results' / 'ep_families2.json'))
FS = ('QWEN_A', 'QWEN_B', 'LLAMA')
EPS = {F: sorted(e[-2:] for e in FAM if FAM[e] == F) for F in FS}
PAIRS = {F: list(itertools.combinations(EPS[F], 2)) for F in FS}
# white box: identified disguises (forced into the group) and detected fp8 (excluded)
COSM = {'QWEN_A': ['02', '23'], 'QWEN_B': ['08', '20'], 'LLAMA': ['16', '29']}
COSM2 = {'QWEN_A': ['04', '33'], 'QWEN_B': ['06', '30'], 'LLAMA': ['10', '31']}   # watermark, T = 1
FP8 = {'QWEN_A': [], 'QWEN_B': ['22'], 'LLAMA': ['15']}
def truths(F):
    must = COSM[F] + ([] if FREE_COSM else COSM2[F]); free = [e for e in EPS[F] if e not in must and e not in FP8[F]]
    out = []
    for m in range(4, 10):
        for extra in itertools.combinations(free, m - len(must)) if m >= len(must) else []:
            out.append(tuple(sorted(must + list(extra))))
    return out
def groups_of(path):
    g = {}
    for r in csv.DictReader(open(path)): g[r['endpoint'][-2:]] = r['model']
    return g
OBS = [l.split() for l in open(T / 'results' / 'submissions' / 'obs.txt') if l.strip() and not l.startswith('#')]
SUBS = [groups_of(T / 'results' / 'submissions' / Path(f).name) for f, _ in OBS]; obs = np.array([float(s) for _, s in OBS])
def cand(A, B, L):
    g = {}
    for F, grp in (('QWEN_A', A), ('QWEN_B', B), ('LLAMA', L)):
        for e in EPS[F]: g[e] = 'M' + F if e in grp else 'x' + e
    return g
CUR = SUBS[-1]
A0 = ['01', '02', '04', '12', '23', '33']; B0 = ['03', '06', '07', '08', '20', '21', '30']; L0 = ['10', '13', '16', '19', '27', '29', '31']
CANDS = {'last submitted (11:17)': CUR, '07:54 (retained best)': cand(A0, B0, [x for x in L0 if x != '19']),
         'A +14': cand(A0 + ['14'], B0, L0),
         'B 26 32 instead of 03 21 (white-box analogy)': cand(A0, ['06', '07', '08', '20', '26', '30', '32'], L0),
         'A +14 and B analogy': cand(A0 + ['14'], ['06', '07', '08', '20', '26', '30', '32'], L0),
         'B +26 +32': cand(A0, B0 + ['26', '32'], L0),
         'B +32': cand(A0, B0 + ['32'], L0), 'B +26': cand(A0, B0 + ['26'], L0),
         'B -21': cand(A0, [x for x in B0 if x != '21'], L0), 'B -03': cand(A0, [x for x in B0 if x != '03'], L0),
         'L -19': cand(A0, B0, [x for x in L0 if x != '19']), 'L -27': cand(A0, B0, [x for x in L0 if x != '27']),
         'L +09': cand(A0, B0, L0 + ['09']),
         'A +17': cand(A0 + ['17'], B0, L0), 'A -12': cand([x for x in A0 if x != '12'], B0, L0),
         'B +32 and L -19': cand(A0, B0 + ['32'], [x for x in L0 if x != '19']),
         'B +32 and L -27': cand(A0, B0 + ['32'], [x for x in L0 if x != '27']),
         'L -19 -27': cand(A0, B0, [x for x in L0 if x not in ('19', '27')]),
         'L +09 -27': cand(A0, B0, [x for x in L0 if x != '27'] + ['09']),
         'B +32 -21': cand(A0, [x for x in B0 if x != '21'] + ['32'], L0),
         'A +18': cand(A0 + ['18'], B0, L0), 'A +24': cand(A0 + ['24'], B0, L0)}
ALL = SUBS + list(CANDS.values()); K = len(ALL); KO = len(SUBS)
rng = np.random.default_rng(0)
def ind(F, tr):
    """(55, K, 4) TP, FP, FN, TN indicators per pair and per submission, for the truth tr (main group)."""
    I = np.zeros((len(PAIRS[F]), K, 4), np.float32)
    for i, (a, b) in enumerate(PAIRS[F]):
        t = a in tr and b in tr
        for k, g in enumerate(ALL):
            pr = g[a] == g[b]; I[i, k, (0 if pr else 2) if t else (1 if pr else 3)] = 1
    return I
S = {F: (rng.random((nS, len(PAIRS[F]))) < p).astype(np.float32) for F in FS}
TR = {F: truths(F) for F in FS}
C = {}; TOT = {}
for F in FS:
    I = np.stack([ind(F, t) for t in TR[F]])                         # (nc, 55, K, 4)
    C[F] = np.einsum('sp,cpkj->cskj', S[F], I)                       # (nc, nS, K, 4) public pairs
    TOT[F] = I.sum(1)                                                # (nc, K, 4)
print('candidate truths:', {F: len(TR[F]) for F in FS}, '| splits:', nS, '| public fraction:', p, '| free disguises:', FREE_COSM)
def score(c):
    tp, fp, fn, tn = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0.0)
    return 0.1 + 0.9 * np.maximum(m, 0)
Wt = 0.0; up = np.zeros(K - KO); uph = np.zeros(K - KO); hid = np.zeros(K - KO); hid2 = np.zeros(K - KO); pub = np.zeros(K - KO)
memb = {F: np.zeros(len(TR[F])) for F in FS}
for a in range(len(TR['QWEN_A'])):
    for b in range(len(TR['QWEN_B'])):
        cp = C['QWEN_A'][a][None] + C['QWEN_B'][b][None] + C['LLAMA']             # (nL, nS, K, 4)
        sp = score(cp[:, :, :KO]); dd = np.maximum(np.abs(sp - obs) - 0.025, 0)          # distance to the rounding interval
        ok = np.exp(-(dd ** 2).sum(-1) / (2 * SIG ** 2))                                   # (nL, nS) vraisemblance souple
        if ok.max() < 1e-12: continue
        ch = (TOT['QWEN_A'][a][None] + TOT['QWEN_B'][b][None] + TOT['LLAMA'])[:, None] - cp   # hidden pairs
        sh = score(ch[:, :, KO:]); spc = score(cp[:, :, KO:])
        w = ok.astype(np.float64); Wt += w.sum()
        hid += np.einsum('ls,lsk->k', w, sh); hid2 += np.einsum('ls,lsk->k', w, sh ** 2); pub += np.einsum('ls,lsk->k', w, spc)
        hi = (np.round(spc / 0.05) * 0.05 > 0.85 + 1e-6).astype(np.float64)
        up += np.einsum('ls,lsk->k', w, hi); uph += np.einsum('ls,lsk->k', w, hi * sh)
        memb['QWEN_A'][a] += w.sum(); memb['QWEN_B'][b] += w.sum(); memb['LLAMA'] += w.sum(1)
print('total weight (soft likelihood, sigma =', SIG, '):', round(Wt, 3))
if Wt == 0: sys.exit()
for F in FS:
    m = {e: 0.0 for e in EPS[F]}
    for t, w in zip(TR[F], memb[F]):
        for e in t: m[e] += w / Wt
    top = sorted(zip(memb[F], TR[F]), reverse=True)[:4]
    print(f'{F:7s} P(main group):', ' '.join(f'{e}:{m[e]:.2f}' for e in EPS[F]))
    print('         most likely truths:', ' ; '.join(f'{" ".join(t)} ({w / Wt:.2f})' for w, t in top))
print('\ncandidate (relative to the 08:24 partition)            exp. public   exp. HIDDEN ± sd     P(public >= 0.90)   hidden if >= 0.90')
mh = hid / Wt; sd = np.sqrt(np.maximum(hid2 / Wt - mh ** 2, 0))
for i in np.argsort(-mh):
    print(f'{list(CANDS)[i][:55]:55s} {pub[i] / Wt:6.3f}          {mh[i]:6.3f} ± {sd[i]:.3f}       {up[i] / Wt:6.3f}            {uph[i] / max(up[i], 1e-9):6.3f}')
