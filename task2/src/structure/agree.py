"""Accord par item entre endpoints (tirages croisés) et accord d'un endpoint avec lui-même (tirages distincts)."""
import itertools, sys
import numpy as np
import parse, plan2

def matrix(A, eps, prefix):
    its = sorted(i for i in A[eps[0]] if i.startswith(prefix)); n = len(eps); M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a, n):
            v = []
            for i in its:
                xa, xb = A[eps[a]][i], A[eps[b]][i]
                if a == b:
                    pr = [x == y for x, y in itertools.combinations(xa, 2) if x != '∅' and y != '∅']
                else:
                    pr = [x == y for x in xa for y in xb if x != '∅' and y != '∅']
                if pr: v.append(np.mean(pr))
            if v: M[a, b] = M[b, a] = np.mean(v)
    return M

def show(M, eps, title):
    print(title); print('      ' + ' '.join(e[-2:].rjust(3) for e in eps))
    for a, e in enumerate(eps): print(e + ' ' + ' '.join('  .' if np.isnan(M[a, b]) else f'{100 * M[a, b]:3.0f}' for b in range(len(eps))))

if __name__ == '__main__':
    D = parse.load()
    for F in sys.argv[1:] or ['QWEN_A', 'QWEN_B', 'LLAMA']:
        eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == F); A = parse.items(D, eps)
        print(f'\n===== {F}  (tirages : frag {len(A[eps[0]]["frag01"])}, tool {len(A[eps[0]]["tool01"])}, tie {len(A[eps[0]]["tie01"])})')
        tot = None
        for k in ['frag', 'tool', 'tie']:
            M = matrix(A, eps, k); show(M, eps, f'-- {k} : accord x100 (diagonale = accord avec soi-même)')
            tot = M if tot is None else tot + M
        show(tot / 3, eps, '-- moyenne des trois batteries')
