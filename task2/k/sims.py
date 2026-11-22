"""Similarités entre endpoints d'une famille, et leur validation sur QWEN_A (dont le découpage a été confirmé par le
diagnostic de 1 h 19). On mesure l'AUC « paires du groupe principal contre toutes les autres paires » pour chaque
similarité, puis on applique LA MÊME similarité aux deux autres familles. Aucun paramètre n'est réglé sur QWEN_B ou LLAMA."""
import collections, itertools, sys
import numpy as np
import parse, plan2, agree, seqstats

QA_MAIN = ['ep-01', 'ep-02', 'ep-04', 'ep-12', 'ep-23', 'ep-33']

def mode_matrix(A, eps, prefix, min_rep=2):
    """Accord des réponses modales (façon Stemma). Un item sans mode net (égalité en tête) ne compte pas pour cet endpoint."""
    its = sorted(i for i in A[eps[0]] if i.startswith(prefix))
    def mode(xs):
        xs = [x for x in xs if x != '∅']
        if len(xs) < min_rep: return None
        c = collections.Counter(xs).most_common(2)
        return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None
    Mo = {e: {i: mode(A[e][i]) for i in its} for e in eps}; n = len(eps); M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            both = [i for i in its if Mo[eps[a]][i] is not None and Mo[eps[b]][i] is not None]
            if len(both) >= 5: M[a, b] = M[b, a] = np.mean([Mo[eps[a]][i] == Mo[eps[b]][i] for i in both])
    return M

def cosine_matrix(A, eps, prefix):
    """Accord croisé normalisé par l'accord avec soi-même : estime le cosinus entre distributions de réponses, item par item."""
    C = agree.matrix(A, eps, prefix); n = len(eps); M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            d = np.sqrt(max(C[a, a], 1e-3) * max(C[b, b], 1e-3)); M[a, b] = M[b, a] = C[a, b] / d
    return M

def auc(M, eps, main):
    pos, neg = [], []
    for a, b in itertools.combinations(range(len(eps)), 2):
        if np.isnan(M[a, b]): continue
        (pos if eps[a] in main and eps[b] in main else neg).append(M[a, b])
    if not pos or not neg: return np.nan
    return np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])

def all_sims(D, eps):
    A = parse.items(D, eps); S = {}
    for k in ['frag', 'tool', 'tie']:
        S[f'brut_{k}'] = agree.matrix(A, eps, k); S[f'mode_{k}'] = mode_matrix(A, eps, k); S[f'cos_{k}'] = cosine_matrix(A, eps, k)
    for kind in ['brut', 'mode', 'cos']:
        S[f'{kind}_tout'] = np.nanmean([S[f'{kind}_{k}'] for k in ['frag', 'tool', 'tie']], axis=0)
    return S, A

if __name__ == '__main__':
    D = parse.load()
    eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == 'QWEN_A'); S, A = all_sims(D, eps)
    print('tirages :', {k: len(A[eps[0]][k + '01']) for k in ['frag', 'tool', 'tie']})
    print('AUC sur QWEN_A (groupe principal confirmé par diagnostic) :')
    for k, M in S.items(): print(f'  {k:10s} {auc(M, eps, QA_MAIN):.3f}')
    core = [e for e in QA_MAIN if e not in ('ep-02', 'ep-23')]
    print('AUC en ne comptant comme positives que les paires hors prompt système/persona :')
    for k, M in S.items(): print(f'  {k:10s} {auc(M, eps, core):.3f}')
