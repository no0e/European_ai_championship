"""Axe dose-effet du LoRA : on projette chaque endpoint du groupe « indiscernable » sur la direction qui mène du groupe
vers les variantes fortes de la même famille. Un LoRA faible doit s'y projeter positivement (même adaptateur, dose
moindre). Erreur type par bootstrap sur les tirages. Validation sur QWEN_A (variantes fortes 18 et 25)."""
import collections, sys
import numpy as np
import parse, plan2

def freq_vectors(A, eps, prefixes=('tool', 'tie'), boot=None, rng=None):
    its = sorted(i for i in A[eps[0]] if i.startswith(prefixes))
    cats = {i: sorted({x for e in eps for x in A[e][i] if x != '∅'}) for i in its}
    V = {}
    for e in eps:
        v = []
        for i in its:
            xs = [x for x in A[e][i] if x != '∅']
            if boot and xs: xs = [xs[j] for j in rng.integers(0, len(xs), len(xs))]
            c = collections.Counter(xs); n = max(len(xs), 1)
            v += [c[k] / n for k in cats[i]]
        V[e] = np.array(v)
    return V

def projections(A, eps, cluster, strong, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    def proj(V):
        out = {}
        for e in eps:
            ref = np.mean([V[c] for c in cluster if c != e], axis=0)
            ax = np.mean([V[s] for s in strong if s != e], axis=0) - ref
            out[e] = float((V[e] - ref) @ ax / (ax @ ax))      # 0 = au niveau du groupe, 1 = au niveau des variantes fortes
        return out
    p0 = proj(freq_vectors(A, eps)); B = [proj(freq_vectors(A, eps, boot=True, rng=rng)) for _ in range(nboot)]
    return {e: (p0[e], float(np.std([b[e] for b in B]))) for e in eps}

if __name__ == '__main__':
    D = parse.load()
    CFG = {'QWEN_A': (['ep-01', 'ep-04', 'ep-12', 'ep-14', 'ep-17', 'ep-33'], ['ep-18', 'ep-25']),
           'LLAMA': (['ep-09', 'ep-13', 'ep-15', 'ep-16', 'ep-19', 'ep-27', 'ep-29', 'ep-31'], None),
           'QWEN_B': (['ep-03', 'ep-07', 'ep-08', 'ep-20', 'ep-21', 'ep-26', 'ep-30', 'ep-32'], None)}
    STRONG = {'LLAMA': [['ep-05', 'ep-11'], ['ep-10'], ['ep-05', 'ep-10', 'ep-11']], 'QWEN_B': [['ep-28'], ['ep-06'], ['ep-22']], 'QWEN_A': [['ep-18', 'ep-25'], ['ep-24']]}
    for F in sys.argv[1:] or ['QWEN_A', 'LLAMA', 'QWEN_B']:
        eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == F); A = parse.items(D, eps); cluster = CFG[F][0]
        for strong in STRONG[F]:
            P = projections(A, eps, cluster, strong)
            print(f'{F} axe vers {[s[-2:] for s in strong]} :', '  '.join(f"{e[-2:]}{'*' if e in cluster else ''}={P[e][0]:+.2f}±{P[e][1]:.2f}" for e in eps))
