"""Cibles « marge par rang » : pour chaque image à oublier, on garde les logits d'orig des AUTRES classes (échelle tb) et on impose
la marge de la classe d'origine m(x) = mu - sigma * Phi^-1(u(x)) - beta * rho_c, où u(x) = rang de difficulté (0 facile, 1 difficile)
et rho_c = part relative de la classe dans le jeu à oublier (n_c / max n_c).
Classements : m0 (marge d'orig sur l'image propre), mv (marge moyenne d'orig sur l'image propre + V vues), oof:<fichier cibles> ; mélange par moyenne de rangs avec '+'.
Usage : python tg.py --env ENV --rank mv --mu 4 --sigma 5 --tb 1 --out tgt/rk.pt     (simulateur : affiche le score des cibles)"""
import argparse, os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3

def margin(l, y):
    o = l.clone(); ar = torch.arange(len(y)); o[ar, y] = -1e9; return l[ar, y] - o.max(1).values
def rank01(d): return (d.argsort().argsort().float() + 0.5) / len(d)

def difficulty(env, spec, z0, y, V=4, cache="rank_cache.pt"):
    us = []
    for s in spec.split("+"):
        if s == "m0": d = -margin(z0, y)
        elif s == "mv":
            if os.path.exists(cache): zs = torch.load(cache)
            else:
                m = env.new_model(); zs = torch.stack([k3.logits(env, m, env.forget, view=v) for v in range(1, V + 1)]); torch.save(zs, cache)
            d = -(margin(z0, y) + sum(margin(zz, y) for zz in zs)) / (len(zs) + 1)
        elif s.startswith("oof:"):
            d = -margin(k3.p2z(torch.load(s[4:])["clean"]), y)
        elif s.startswith("file:"): d = torch.load(s[5:])["u"]          # classement externe (k4/combine.py), 1 = difficile
        elif s == "rand": d = torch.rand(len(y), generator=torch.Generator().manual_seed(0))
        else: raise ValueError(s)
        us.append(rank01(d))
    return rank01(sum(us)) if len(us) > 1 else us[0]

def neighbor_bias(sig_file, layer, gamma, alpha=0.5):
    """Biais sur les AUTRES classes d'après les classes des 20 plus proches voisins (features d'orig à mi-profondeur) :
    vers quelle mauvaise classe un modèle qui n'a pas mémorisé l'image pencherait. Nul pour les classes absentes du jeu à oublier."""
    d = torch.load(sig_file); h = d["hist"][layer]; cls = d["classes"]; N = len(h)
    lp = torch.log((h + alpha) / (h.sum(1, keepdim=True) + alpha * h.shape[1])); lp = lp - lp.mean(1, keepdim=True)
    ob = torch.zeros(N, d["z0"].shape[1]); ob[:, cls] = gamma * lp
    return ob


def build(z0, y, u, mu, sigma, tb=1.0, beta=0.0, shape="normal", obias=None, tb_neg=None, floor=None):
    """tb_neg : échelle des autres classes pour les images visées FAUSSES (marge < 0), si différente de tb ; floor : marge minimale."""
    ar = torch.arange(len(y)); o = z0.clone().float(); o[ar, y] = -1e9; mx = o.max(1, keepdim=True).values
    cnt_ = y.bincount(minlength=z0.shape[1]).float(); rho_ = (cnt_ / cnt_.max())[y]
    mh_ = mu - sigma * torch.distributions.Normal(0, 1).icdf(u.clamp(1e-4, 1 - 1e-4)) - beta * rho_
    if floor is not None: mh_ = mh_.clamp_min(-floor)
    tbv = torch.full((len(y),), float(tb))
    if tb_neg is not None: tbv = torch.where(mh_ < 0, torch.full_like(tbv, float(tb_neg)), tbv)
    build.tbv = tbv; build.floor = floor
    t = mx + (z0.float() - mx) * tbv[:, None]
    if obias is not None:
        t = t + obias; o2 = t.clone(); o2[ar, y] = -1e9; mx = o2.max(1, keepdim=True).values      # la marge se mesure après le biais
    cnt = y.bincount(minlength=z0.shape[1]).float(); rho = (cnt / cnt.max())[y]
    q = torch.distributions.Normal(0, 1).icdf(u.clamp(1e-4, 1 - 1e-4))
    mhat = mu - sigma * q - beta * rho
    if floor is not None: mhat = mhat.clamp_min(-floor)
    ps = getattr(build, "pos_scale", 1.0)
    if ps != 1.0: mhat = torch.where(mhat > 0, mhat * ps, mhat)        # images visées justes : marges resserrées (<1) ou élargies (>1)
    ns = getattr(build, "neg_scale", 1.0)
    if ns != 1.0: mhat = torch.where(mhat < 0, mhat * ns, mhat)        # images visées fausses : erreurs moins (<1) ou plus (>1) profondes
    t[ar, y] = mx[:, 0] + mhat
    return t, mhat

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="real"); ap.add_argument("--rank", default="mv"); ap.add_argument("--out", default="tgt/rk.pt")
    ap.add_argument("--mu", type=float, default=4); ap.add_argument("--sigma", type=float, default=5); ap.add_argument("--tb", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--tb_neg", type=float, default=None); ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--pos_scale", type=float, default=1.0); ap.add_argument("--neg_scale", type=float, default=1.0)
    ap.add_argument("--bulk_k", type=int, default=0); ap.add_argument("--bulk_d", type=float, default=0.0)
    ap.add_argument("--pi_gamma", type=float, default=0.0, help="biais vers le profil de confusion de la classe (mauvaises classes typiques de la classe), en logits")
    ap.add_argument("--obias", default="", help="fichier_signaux:couche:gamma, ex. ../k4/sig2_real.pt:d62:1.0")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.env == "real" else a.out
    env = k3.get_env(a.env, "cuda")
    z0 = k3.logits(env, env.new_model(), env.forget); y = z0.argmax(1)
    u = difficulty(env, a.rank, z0, y)
    ob = None
    if a.pi_gamma:                      # profil de confusion par classe : moyenne, sur les images de la classe, du softmax des AUTRES classes d'orig
        ar_ = torch.arange(len(y)); oth = z0.clone().float(); oth[ar_, y] = -1e9; so = F.softmax(oth / 2, 1)
        ob = torch.zeros_like(z0, dtype=torch.float)
        for c in y.unique():
            lp = torch.log(so[y == c].mean(0).clamp_min(1e-6)); lp[c] = lp[lp > -13].mean(); ob[y == c] = a.pi_gamma * (lp - lp.max())
        ob[ar_, y] = 0
    if a.obias:
        f_, l_, g_ = a.obias.rsplit(":", 2); ob = neighbor_bias(f_, l_, float(g_))
    build.pos_scale = a.pos_scale; build.neg_scale = a.neg_scale
    t, mhat = build(z0, y, u, a.mu, a.sigma, a.tb, a.beta, obias=ob, tb_neg=a.tb_neg, floor=a.floor)
    ce = F.cross_entropy(t, y, reduction="none")
    print(f"cibles {a.rank} mu={a.mu} sigma={a.sigma} tb={a.tb} beta={a.beta}: acc {(t.argmax(1) == y).float().mean():.3f} CE {ce.mean():.2f}±{ce.std():.2f} p(y)@T2 {F.softmax(t / 2, 1)[torch.arange(len(y)), y].mean():.3f}")
    if a.bulk_k:                        # même règle que k3.cap_targets, pour que les statistiques affichées soient celles des cibles réelles
        k3.BULK = (a.bulk_k, a.bulk_d); t, _ = k3.cap_targets(z0.float(), y, mhat, a.tb, ob)
    tgt = dict(clean=F.softmax(t / 2, 1), views=None, mhat=mhat, tb=a.tb, obias=ob, tbv=build.tbv if a.tb_neg is not None else None,
               bulk=(a.bulk_k, a.bulk_d) if a.bulk_k else None)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); torch.save(tgt, out); print("OK", out)
    if env.sim: k3.eval_targets(env, tgt, a.rank, grid=False)
