"""Toy Model Equality Testing: MMD with the Hamming kernel and a permutation p-value (numpy only)."""
import numpy as np

def mmd_hamming(X, Y, px, py):
    """X: (n, L) completions from P, Y: (n, L) from Q; px, py: prompt indices."""
    L = X.shape[1]
    def K(A, B, pa, pb):
        k = (A[:, None, :] == B[None, :, :]).sum(-1) / L   # normalised Hamming kernel
        mask = pa[:, None] == pb[None, :]                    # 1{x = x'}
        return k * mask, mask
    Kxx, Mxx = K(X, X, px, px); Kyy, Myy = K(Y, Y, py, py); Kxy, Mxy = K(X, Y, px, py)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Mxx, False)
    np.fill_diagonal(Kyy, 0); np.fill_diagonal(Myy, False)
    return Kxx.sum()/Mxx.sum() + Kyy.sum()/Myy.sum() - 2*Kxy.sum()/Mxy.sum()

def permutation_pvalue(X, Y, px, py, B=200, rng=None):
    rng = rng or np.random.default_rng()
    obs = mmd_hamming(X, Y, px, py)
    Z, pz, n = np.vstack([X, Y]), np.concatenate([px, py]), len(X)
    stats = []
    for _ in range(B):
        i = rng.permutation(len(Z))
        stats.append(mmd_hamming(Z[i[:n]], Z[i[n:]], pz[i[:n]], pz[i[n:]]))
    return np.mean(np.array(stats) >= obs), obs

# ---- A toy "mini-LLM": one bigram model per prompt ----
V, L, m = 30, 20, 5
rng0 = np.random.default_rng(0)
logits = rng0.normal(0, 2.0, size=(m, V, V))     # logits[prompt, previous_token, next_token]
bruit  = rng0.normal(0, 1.0, size=(m, V, V))     # fixed perturbation (simulates quantisation)

def sample(n_per_prompt, temperature=1.0, noise=0.0, rng=None):
    rng = rng or np.random.default_rng()
    lg = logits + noise * bruit
    X, p = [], []
    for x in range(m):
        for _ in range(n_per_prompt):
            tok, seq = x % V, []
            for _ in range(L):
                z = lg[x, tok] / temperature
                pr = np.exp(z - z.max()); pr /= pr.sum()
                tok = rng.choice(V, p=pr); seq.append(tok)
            X.append(seq); p.append(x)
    return np.array(X), np.array(p)

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for name, kw in [("P vs P (H0 true)", {}),
                     ("P vs temperature 0.8", {"temperature": 0.8}),
                     ("P vs temperature 0.6", {"temperature": 0.6}),
                     ("P vs 'quantised' (noise 0.5)", {"noise": 0.5}),
                     ("P vs 'quantised' (noise 1.0)", {"noise": 1.0})]:
        rej = []
        for s in range(40):
            X, px = sample(20, rng=rng)
            Y, py = sample(20, rng=rng, **kw)
            pval, stat = permutation_pvalue(X, Y, px, py, B=100, rng=rng)
            rej.append(pval < 0.05)
        print(f"{name:32s} stat(last)={stat:+.4f}  rejection rate={np.mean(rej):.2f}")
