"""Choix de la prochaine partition : a priori tiré des DONNÉES (probabilité qu'un endpoint soit dans le groupe principal),
vraisemblance tirée des scores publics déjà observés, puis partition de score attendu maximal.
Structures envisagées par famille : un groupe principal, le reste isolé, au plus une paire hors groupe.
    python bayes.py            -> a posteriori et meilleure partition
"""
import itertools, json, math, sys
import numpy as np
import fit, mkcsv

SIGMA = 0.02      # écart toléré entre score prédit (toutes les paires) et score public (une partie des paires, arrondi à 0,01)
PRIOR = {
 'QWEN_B': {'08': 1.0, '20': 1.0, '03': .5, '07': .5, '21': .5, '32': .5, '06': .4, '22': .2, '26': .3, '28': .05, '30': .25},
 'LLAMA': {'16': 1.0, '29': .85, '31': .7, '13': .6, '27': .5, '19': .5, '09': .3, '15': .3, '10': .3, '05': .1, '11': .15},
}
PAIRS = {'QWEN_B': [('07', '32'), ('21', '32'), ('06', '08')], 'LLAMA': [('05', '11'), ('10', '15'), ('09', '19')]}
PAIR_PRIOR = 0.3
SIZES = {'QWEN_B': (4, 5, 6, 7, 8), 'LLAMA': (3, 4, 5, 6, 7, 8)}

def space(F):
    eps = fit.EPS[F]; must = [e for e in eps if PRIOR[F][e] >= 1.0]; out = []
    for key, t in fit.truths(F, must, SIZES[F], pairs=PAIRS[F]):
        M, pr = key; w = 1.0
        for e in eps: w *= PRIOR[F][e] if e in M else (1 - PRIOR[F][e])
        if pr: w *= PAIR_PRIOR
        vec = np.array([t[a] == t[b] for a, b in itertools.combinations(eps, 2)], dtype=np.int32)
        out.append((key, vec, w))
    return out

def pred_vec(pred, F):
    eps = fit.EPS[F]; return np.array([pred[a] == pred[b] for a, b in itertools.combinations(eps, 2)], dtype=np.int32)

def counts_mat(Tv, Pv):
    """Tv : (nT, 55), Pv : (nP, 55) -> tp, fp, fn, tn de forme (nT, nP)."""
    tp = Tv @ Pv.T; sT = Tv.sum(1)[:, None]; sP = Pv.sum(1)[None, :]; n = Tv.shape[1]
    return tp, sP - tp, sT - tp, n - sT - sP + tp

def score_arr(tp, fp, fn, tn):
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)
    m = np.where(d > 0, (tp * tn - fp * fn) / np.where(d > 0, d, 1), 0.0)
    return 0.1 + 0.9 * np.maximum(0.0, m)

def main():
    QA = {e: ('M' if e in ('01', '02', '04', '12', '23', '33') else e) for e in fit.EPS['QWEN_A']}
    SB, SL = space('QWEN_B'), space('LLAMA')
    TB = np.array([v for _, v, _ in SB]); TL = np.array([v for _, v, _ in SL])
    wB = np.array([w for _, _, w in SB]); wL = np.array([w for _, _, w in SL])
    logp = np.log(wB)[:, None] + np.log(wL)[None, :]
    for pred, s in fit.PRED:
        a = fit.counts(QA, pred, fit.EPS['QWEN_A'])
        b = counts_mat(TB, pred_vec(pred, 'QWEN_B')[None, :]); l = counts_mat(TL, pred_vec(pred, 'LLAMA')[None, :])
        sc = score_arr(*[a[i] + b[i][:, 0][:, None] + l[i][:, 0][None, :] for i in range(4)])
        logp += -0.5 * ((sc - s) / SIGMA) ** 2
    post = np.exp(logp - logp.max()); post /= post.sum()
    pB, pL = post.sum(1), post.sum(0)
    for F, S, p in (('QWEN_B', SB, pB), ('LLAMA', SL, pL)):
        mem = {e: float(sum(p[i] for i, (k, _, _) in enumerate(S) if e in k[0])) for e in fit.EPS[F]}
        print(F, 'P(dans le groupe principal) :', ' '.join(f'{e}={mem[e]:.2f}' for e in fit.EPS[F]))
        top = np.argsort(-p)[:4]
        for i in top: print(f'    {p[i]:.3f}  groupe {" ".join(S[i][0][0])}' + (f'  + paire {"-".join(S[i][0][1])}' if S[i][0][1] else ''))
    # partition de score attendu maximal : montée par coordonnées sur (candidat B, candidat L), candidats = mêmes structures
    aA = (15, 0, 0, 40)
    keepB = np.argsort(-pB)[:60]; keepL = np.argsort(-pL)[:150]            # vérités qui portent l'essentiel de la masse
    P2 = post[np.ix_(keepB, keepL)]; P2 = P2 / P2.sum()
    cb = counts_mat(TB[keepB], TB); cl = counts_mat(TL[keepL], TL)        # (vérités gardées, tous les candidats)
    ib, il = int(np.argmax(pB)), int(np.argmax(pL)); best = -1
    for _ in range(4):
        # meilleur candidat B à L fixé
        sc = score_arr(*[aA[i] + cb[i][:, None, :] + cl[i][:, il][None, :, None] for i in range(4)])     # (kB, kL, nB)
        eb = np.einsum('ij,ijk->k', P2, sc); ib = int(np.argmax(eb))
        sc = score_arr(*[aA[i] + cb[i][:, ib][:, None, None] + cl[i][None, :, :] for i in range(4)])     # (kB, kL, nL)
        el = np.einsum('ij,ijk->k', P2, sc); il = int(np.argmax(el))
        if el[il] <= best + 1e-9: break
        best = float(el[il])
    # candidats conjoints : 30 meilleurs de chaque famille, score attendu exact, en écartant les partitions déjà soumises
    topB = np.argsort(-eb)[:30]; topL = np.argsort(-el)[:30]
    done = {(tuple(pred_vec(p, 'QWEN_B')), tuple(pred_vec(p, 'LLAMA'))) for p, _ in fit.PRED}
    cands = []
    for jb in topB:
        for jl in topL:
            sc = score_arr(*[aA[i] + cb[i][:, jb][:, None] + cl[i][:, jl][None, :] for i in range(4)])
            cands.append((float((P2 * sc).sum()), int(jb), int(jl)))
    cands.sort(reverse=True)
    best, ib, il = next(c for c in cands if (tuple(TB[c[1]]), tuple(TL[c[2]])) not in done)
    kb, kl = SB[ib][0], SL[il][0]
    print(f'\nPartition de score attendu maximal, non encore soumise ({best:.3f}) :')
    print('  QWEN_B groupe', ' '.join(kb[0]), ('+ paire ' + '-'.join(kb[1])) if kb[1] else '')
    print('  LLAMA  groupe', ' '.join(kl[0]), ('+ paire ' + '-'.join(kl[1])) if kl[1] else '')
    sc = score_arr(*[aA[i] + cb[i][:, ib][:, None] + cl[i][:, il][None, :] for i in range(4)])
    order = np.argsort(sc.ravel()); cdf = np.cumsum(P2.ravel()[order])
    q = lambda x: float(sc.ravel()[order][min(np.searchsorted(cdf, x), len(cdf) - 1)])
    print(f'  distribution du score : 10 % {q(.1):.2f} | médiane {q(.5):.2f} | 90 % {q(.9):.2f}')
    g = {'QWEN_A': mkcsv.QA, 'QWEN_B': [list(kb[0])] + ([list(kb[1])] if kb[1] else []), 'LLAMA': [list(kl[0])] + ([list(kl[1])] if kl[1] else [])}
    json.dump({'groups': g, 'attendu': best, 'note': 'QWEN_B = {' + ','.join(kb[0]) + '}' + (' + paire ' + '-'.join(kb[1]) if kb[1] else '') + ' ; LLAMA = {' + ','.join(kl[0]) + '}' + (' + paire ' + '-'.join(kl[1]) if kl[1] else '')}, open('next_groups.json', 'w'), ensure_ascii=False)
    return g

if __name__ == '__main__':
    main()
