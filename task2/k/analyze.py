"""Analyse des réponses longues, famille par famille. Aucune étiquette, aucun score public.
   python analyze.py [QWEN_A|QWEN_B|LLAMA] [--perm 400]
Sorties : profil par endpoint (entropie, respect du format, longueurs) et matrice des tests à deux échantillons
(MMD à noyau delta par item, façon AgentProv ; permutation des RÉPONSES ENTIÈRES car les items d'une réponse sont liés).
"""
import collections, math, sys
import numpy as np
import parse, plan2

def profile(D, A, e):
    ent, nul = [], []
    for it, xs in A[e].items():
        c = collections.Counter(xs); n = len(xs)
        ent.append(1 - sum((v / n) ** 2 for v in c.values()) * n / max(n - 1, 1) + 1 / max(n - 1, 1) if n > 1 else np.nan)
        nul.append(np.mean([x == '∅' for x in xs]))
    coin = [len(parse.coin(t)) for t in D['seq_coin'].get(e, [])]
    dig = [len(parse.digits(t)) for t in D['seq_digits'].get(e, [])]
    long = [len(t) for p in D if p.startswith('off') for t in D[p].get(e, [])]
    return dict(n=min((len(v) for v in A[e].values()), default=0), divers=np.nanmean(ent) if ent else np.nan, vide=np.mean(nul) if nul else np.nan,
                coin=np.mean(coin) if coin else np.nan, dig=np.mean(dig) if dig else np.nan,
                loglen=np.mean([math.log(1 + x) for x in long]) if long else np.nan)

def resp_vectors(D, e, kinds=('bat_frag', 'bat_tool', 'bat_tie')):
    """Par type de batterie : liste de réponses, chacune = dict item -> réponse."""
    out = {}
    for k in kinds:
        key = 'bat_tool_l' if (k == 'bat_tool' and plan2.FAM.get(e) == 'LLAMA') else k
        rs = []
        for txt in D[key].get(e, []):
            if k == 'bat_tool': b = parse.tools(txt); n = 20
            else: b = parse.battery(txt, 22 if k == 'bat_frag' else 24); n = 22 if k == 'bat_frag' else 24
            rs.append({i: b.get(i, '∅') for i in range(1, n + 1)})
        out[k] = rs
    return out

def mmd_delta(Ra, Rb):
    """Somme sur les items de ||phi_a - phi_b||^2 (fréquences des réponses), estimateur sans biais des termes intra."""
    t = 0.0
    for i in Ra[0]:
        xa = [r[i] for r in Ra if r[i] != '∅']; xb = [r[i] for r in Rb if r[i] != '∅']
        if len(xa) < 2 or len(xb) < 2: continue
        ca, cb = collections.Counter(xa), collections.Counter(xb); na, nb = len(xa), len(xb)
        kaa = (sum(v * v for v in ca.values()) - na) / (na * (na - 1)) if na > 1 else 0
        kbb = (sum(v * v for v in cb.values()) - nb) / (nb * (nb - 1)) if nb > 1 else 0
        kab = sum(ca[k] * cb.get(k, 0) for k in ca) / (na * nb)
        t += kaa + kbb - 2 * kab
    return t

def pair_test(Va, Vb, nperm=400, rng=None):
    rng = rng or np.random.default_rng(0)
    obs = 0.0; perm = np.zeros(nperm); used = 0
    for k in Va:
        Ra, Rb = Va[k], Vb[k]
        if len(Ra) < 2 or len(Rb) < 2: continue
        used += 1; obs += mmd_delta(Ra, Rb); pool = Ra + Rb; na = len(Ra)
        for t in range(nperm):
            q = rng.permutation(len(pool)); perm[t] += mmd_delta([pool[j] for j in q[:na]], [pool[j] for j in q[na:]])
    if not used: return np.nan, np.nan
    return (obs - perm.mean()) / (perm.std() + 1e-12), (1 + (perm >= obs).sum()) / (nperm + 1)

def main():
    fams = [a for a in sys.argv[1:] if not a.startswith('--') and not a.isdigit()] or ['QWEN_A', 'QWEN_B', 'LLAMA']
    nperm = int(sys.argv[sys.argv.index('--perm') + 1]) if '--perm' in sys.argv else 300
    D = parse.load()
    for F in fams:
        eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == F)
        A = parse.items(D, eps)
        print(f'\n== {F}  profil (n = tirages par batterie ; divers = 1 - accord entre tirages ; vide = items sans réponse)')
        for e in eps:
            p = profile(D, A, e)
            print(f"{e} n={p['n']}  divers={p['divers']:.2f}  vide={p['vide']:.2f}  pile/face={p['coin']:.0f}  chiffres={p['dig']:.0f}  loglen_off={p['loglen']:.2f}")
        V = {e: resp_vectors(D, e) for e in eps}; n = len(eps); Z = np.full((n, n), np.nan)
        rng = np.random.default_rng(0)
        for a in range(n):
            for b in range(a + 1, n):
                Z[a, b] = Z[b, a] = pair_test(V[eps[a]], V[eps[b]], nperm, rng)[0]
        print('z du test à deux échantillons (batteries) :'); print('      ' + ' '.join(e[-2:].rjust(5) for e in eps))
        for a in range(n): print(eps[a] + ' ' + ' '.join(('    .' if a == b else f'{Z[a, b]:5.1f}') for b in range(n)))
        np.save(f'z_{F}.npy', Z)

if __name__ == '__main__':
    main()
