"""Tests par paires sur la collecte complète : z du MMD à noyau delta par batterie (permutation des réponses entières)
et taille d'effet (MMD² sans biais). Validation : AUC sur QWEN_A."""
import sys, itertools
import numpy as np
import parse, plan2, analyze, sims, agree

def run(F, kinds=('bat_tool', 'bat_tie', 'bat_frag'), nperm=300, seed=0):
    D = parse.load(); eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == F); n = len(eps)
    V = {e: analyze.resp_vectors(D, e) for e in eps}; rng = np.random.default_rng(seed)
    Z = {k: np.full((n, n), np.nan) for k in kinds}; E = {k: np.full((n, n), np.nan) for k in kinds}
    for k in kinds:
        for a, b in itertools.combinations(range(n), 2):
            Ra, Rb = V[eps[a]][k], V[eps[b]][k]
            z, p = analyze.pair_test({k: Ra}, {k: Rb}, nperm, rng)
            Z[k][a, b] = Z[k][b, a] = z; E[k][a, b] = E[k][b, a] = analyze.mmd_delta(Ra, Rb) / len(Ra[0])
    return eps, Z, E

if __name__ == '__main__':
    for F in sys.argv[1:]:
        eps, Z, E = run(F)
        tot = sum(np.nan_to_num(Z[k]) for k in Z) / np.sqrt(len(Z))
        for k in list(Z) + ['tout']:
            M = tot if k == 'tout' else Z[k]
            print(f'\n== {F} z ({k})'); print('      ' + ' '.join(e[-2:].rjust(5) for e in eps))
            for a in range(len(eps)): print(eps[a] + ' ' + ' '.join(('    .' if a == b else f'{M[a, b]:5.1f}') for b in range(len(eps))))
            if F == 'QWEN_A': print('   AUC groupe / noyau :', round(float(sims.auc(-M, eps, sims.QA_MAIN)), 2), round(float(sims.auc(-M, eps, ['ep-01', 'ep-04', 'ep-12', 'ep-33'])), 2))
        np.save(f'zfull_{F}.npy', np.array([Z[k] for k in Z]))
