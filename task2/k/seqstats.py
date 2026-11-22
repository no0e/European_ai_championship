"""Statistiques de chaîne sur les suites aléatoires (façon IRIS) et forme des réponses, par endpoint."""
import collections, sys
import numpy as np
import parse, plan2

def coin_stats(seqs):
    L = [len(s) for s in seqs]; s = [c for q in seqs for c in q]
    if len(s) < 8: return dict(n=len(seqs), len=np.mean(L) if L else 0, pH=np.nan, alt=np.nan, hhh=np.nan)
    alt = np.mean([a != b for q in seqs for a, b in zip(q, q[1:])])
    tri = collections.Counter(''.join(q[i:i + 3]) for q in seqs for i in range(len(q) - 2)); T = sum(tri.values())
    return dict(n=len(seqs), len=np.mean(L), pH=np.mean([c == 'H' for c in s]), alt=alt, hhh=(tri['HHH'] + tri['TTT']) / max(T, 1))

def digit_stats(seqs):
    s = [c for q in seqs for c in q]
    if len(s) < 10: return dict(len=0, top='', ent=np.nan, rep=np.nan)
    c = collections.Counter(s); p = np.array([c[str(d)] for d in range(10)], float) / len(s)
    rep = np.mean([a == b for q in seqs for a, b in zip(q, q[1:])])
    return dict(len=np.mean([len(q) for q in seqs]), top=''.join(d for d, _ in c.most_common(3)), ent=float(-(p[p > 0] * np.log2(p[p > 0])).sum()), rep=rep)

if __name__ == '__main__':
    D = parse.load()
    for F in sys.argv[1:] or ['QWEN_A', 'QWEN_B', 'LLAMA']:
        eps = sorted(e for e in plan2.FAM if plan2.FAM[e] == F)
        print(f'\n== {F} : pile/face (n, longueur, P(H), alternance, triplets égaux) | chiffres (longueur, top3, entropie, répétition) | formes')
        for e in eps:
            cs = coin_stats([parse.coin(t) for t in D['seq_coin'].get(e, [])]); ds = digit_stats([parse.digits(t) for t in D['seq_digits'].get(e, [])])
            fc = ''.join(str(parse.seq_format(t)) for t in D['seq_coin'].get(e, [])); fd = ''.join(str(parse.seq_format(t)) for t in D['seq_digits'].get(e, []))
            fl = ','.join(str(t.count('\n') + 1) for t in D['bat_frag'].get(e, []))
            print(f"{e} {cs['n']} {cs['len']:4.0f} {cs['pH']:.2f} {cs['alt']:.2f} {cs['hhh']:.2f} | {ds['len']:4.0f} {ds['top']:3s} {ds['ent']:.2f} {ds['rep']:.2f} | coin {fc} dig {fd} | lignes frag {fl}")
