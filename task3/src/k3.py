"""Tâche 3 — pipeline « k3 » : cibles par image estimées hors-pli, puis distillation. Même code pour le VRAI problème
et pour le simulateur (où l'on connaît les golds, donc le barème exact).

  python k3.py fold    --env ENV --k 4 --fold 0 --rep 0 [mécanisme] --out oof/x_r0_f0.pt
  python k3.py merge   --env ENV --inputs "oof/x_r*_f*.pt" --out tgt/x.pt          (moyenne des probas à T=2, par image et par vue)
  python k3.py evalt   --env ENV --targets tgt/x.pt                                 (simulateur : note les cibles elles-mêmes)
  python k3.py distill --env ENV --targets tgt/x.pt [--s 1 --b 0 ...] --out sub/x.pt

ENV = "real" (dossier unlearning/ : task_template.py, model.pt, forget_images.npz, anchors.npz) ou un dossier de simulateur (sim/C).
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F

T = 2.0
AMP = torch.float16 if os.environ.get("K3_AMP") == "fp16" else torch.bfloat16      # fp16 + GradScaler : GPU sans bf16 natif (nœud de connexion)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("T3_WORK", os.path.dirname(HERE)))   # working dir: model.pt, forget_images.npz, anchors.npz, k5/


def kl_t(t_logits, s_logits, T=T):
    lp = F.log_softmax(t_logits.float() / T, 1); lq = F.log_softmax(s_logits.float() / T, 1)
    return (lp.exp() * (lp - lq)).sum(1)


# ----------------------------------------------------------------------------- environnements
def aug_real(x, seeds):
    """Altérations plausibles en 224 px (crop+resize, flip, couleurs, flou, bruit), déterministes par graine. x [B,3,H,W] dans [0,1]."""
    B, dev = x.shape[0], x.device
    out = torch.empty_like(x); gdev = torch.Generator(device=dev); S = x.shape[-1]
    for i in range(B):
        g = torch.Generator().manual_seed(int(seeds[i])); r = torch.rand(12, generator=g).tolist()
        s = 0.6 + 0.4 * r[0]; ratio = math.exp(math.log(3 / 4) + r[1] * (math.log(4 / 3) - math.log(3 / 4)))
        w = max(8, min(S, int(round(S * math.sqrt(s * ratio))))); h = max(8, min(S, int(round(S * math.sqrt(s / ratio)))))
        x0, y0 = min(int(r[2] * (S - w + 1)), S - w), min(int(r[3] * (S - h + 1)), S - h)
        im = F.interpolate(x[i:i + 1, :, y0:y0 + h, x0:x0 + w], size=(S, S), mode="bilinear", align_corners=False)[0]
        if r[4] < 0.5: im = im.flip(2)
        im = im * (1 + (2 * r[5] - 1) * 0.2)
        m = im.mean(); im = (im - m) * (1 + (2 * r[6] - 1) * 0.2) + m
        gray = im.mean(0, keepdim=True); im = (im - gray) * (1 + (2 * r[7] - 1) * 0.2) + gray
        if r[8] < 0.2: im = F.avg_pool2d(F.pad(im[None], (1, 1, 1, 1), mode="reflect"), 3, stride=1)[0]
        gdev.manual_seed(int(seeds[i])); im = im + torch.randn(im.shape, generator=gdev, device=dev) * 0.02 * r[9]
        out[i] = im
    return out.clamp(0, 1)


def aug_strong(x, seeds, smin=0.08, smax=0.45):
    """Vue forte : petit crop (8-45 % de la surface) agrandi + flip + couleurs. Sert de « même classe, image non reconnue »."""
    B = x.shape[0]; out = torch.empty_like(x); S = x.shape[-1]
    for i in range(B):
        g = torch.Generator().manual_seed(int(seeds[i]) + 7919); r = torch.rand(8, generator=g).tolist()
        k = max(4, int(round(S * math.sqrt(smin + (smax - smin) * r[0])))); oy, ox = int(r[1] * (S - k + 1)), int(r[2] * (S - k + 1))
        im = F.interpolate(x[i:i + 1, :, oy:oy + k, ox:ox + k], size=(S, S), mode="bilinear", align_corners=False)[0]
        if r[3] < 0.5: im = im.flip(2)
        m = im.mean(); im = ((im - m) * (1 + (2 * r[4] - 1) * 0.3) + m) * (1 + (2 * r[5] - 1) * 0.3)
        out[i] = im
    return out.clamp(0, 1)


def aug_broad(x, seeds):
    """Famille d'altérations LARGE (les copies altérées du serveur sont inconnues) : celles de aug_real, puis au hasard rotation, flou,
    perte de résolution, niveaux de gris, bruit fort, occlusion, luminosité/contraste marqués, quantification."""
    out = aug_real(x, seeds); B, dev, S = x.shape[0], x.device, x.shape[-1]
    for i in range(B):
        g = torch.Generator().manual_seed(int(seeds[i]) * 7 + 13); r = torch.rand(24, generator=g).tolist(); im = out[i:i + 1]
        if r[0] < 0.35:                                   # rotation +-20 degrés
            th = math.radians((2 * r[1] - 1) * 20); c, sn = math.cos(th), math.sin(th)
            A = torch.tensor([[[c, -sn, 0.0], [sn, c, 0.0]]], device=dev)
            im = F.grid_sample(im, F.affine_grid(A, im.shape, align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
        if r[2] < 0.3:                                    # flou
            k = 3 + 2 * int(r[3] * 3); im = F.avg_pool2d(F.pad(im, (k // 2,) * 4, mode="reflect"), k, stride=1)
        if r[4] < 0.25:                                   # perte de résolution
            q = int(S * (0.35 + 0.45 * r[5])); im = F.interpolate(F.interpolate(im, size=(q, q), mode="bilinear", align_corners=False), size=(S, S), mode="bilinear", align_corners=False)
        if r[6] < 0.1: im = im.mean(1, keepdim=True).expand(-1, 3, -1, -1)                    # niveaux de gris
        if r[7] < 0.3:                                    # luminosité / contraste marqués
            m_ = im.mean(); im = ((im - m_) * (1 + (2 * r[8] - 1) * 0.4) + m_) * (1 + (2 * r[9] - 1) * 0.4)
        if r[10] < 0.3: im = im + 0.08 * r[11] * torch.randn(im.shape, generator=g).to(dev)   # bruit fort
        if r[12] < 0.2:                                   # occlusion
            h_, w_ = int(S * (0.1 + 0.25 * r[13])), int(S * (0.1 + 0.25 * r[14])); y0, x0 = int(r[15] * (S - h_)), int(r[16] * (S - w_))
            im = im.clone(); im[:, :, y0:y0 + h_, x0:x0 + w_] = im.mean()
        if r[17] < 0.2: im = torch.round(im.clamp(0, 1) * 31) / 31                             # quantification
        out[i] = im[0]
    return out.clamp(0, 1)


class Env:
    """forget / anchors : uint8 [N,H,W,3] (CPU). to01 -> float [B,3,H,W] ; aug(x01, seeds) ; inp(x01) -> entrée du modèle."""
    def to01(self, xu8): return xu8.to(self.dev).permute(0, 3, 1, 2).float() / 255.0
    def views(self, xu8, idx, v):
        """v[i] = 0 : image propre ; > 0 : vue altérée déterministe n° v[i] de l'image idx[i] ; < 0 : vue FORTE n° -v[i]."""
        x = self.to01(xu8); sel = [i for i, vv in enumerate(v) if vv > 0]; st = [i for i, vv in enumerate(v) if vv < 0]
        if sel: x[sel] = self.aug(x[sel], [int(idx[i]) * 1000 + int(v[i]) for i in sel])
        if st: x[st] = aug_strong(x[st], [int(idx[i]) * 1000 - int(v[i]) for i in st])
        return x
    def new_model(self):
        m = self.build(); m.load_state_dict(self.orig_sd, strict=True); return m.to(self.dev).eval()


class RealEnv(Env):
    name = "real"
    def __init__(self, dev):
        self.dev = dev; os.chdir(ROOT); sys.path.insert(0, ROOT)
        import task_template as tt
        tt.DEVICE = dev; self.tt = tt; self.build = tt.build_model
        self.orig_sd = torch.load("model.pt", map_location="cpu", weights_only=True)
        self.forget = tt.load_forget_images()
        with np.load("anchors.npz") as d: self.anchors = torch.from_numpy(d["images"].copy())
        self.sim = None; self.anchor_sets = dict(coco=self.anchors)
        if os.path.exists("k5/syn_s0.8.npz"):      # variantes fortes : contenu différent, même style d'image générée -> ancres lointaines « dans le style »
            with np.load("k5/syn_s0.8.npz") as d: self.anchor_sets["syn8"] = torch.from_numpy(d["images"].copy())
        self.syn = None; syn = os.environ.get("K3_SYN", "k5/syn_s0.5.npz")     # variantes synthétiques (diffusion image-vers-image) des images à oublier
        if os.path.exists(syn):
            with np.load(syn) as d: self.syn = torch.from_numpy(d["images"].copy()); src = torch.from_numpy(d["src"].astype("int64"))
            self.anchor_sets["syn"] = self.syn
            order = src.argsort(stable=True); cnt = src.bincount(minlength=len(self.forget)); K = int(cnt.min())
            self.syn_of = order.view(-1, int(cnt.max()))[:, :K] if cnt.min() == cnt.max() else None      # [N, K] lignes des variantes de chaque image
    aug = staticmethod(aug_broad if os.environ.get("K3_AUG") == "broad" else aug_real)
    def inp(self, x01): return (x01 - 0.5) / 0.5


class SimEnv(Env):
    def __init__(self, d, dev):
        self.dev = dev; self.name = os.path.basename(os.path.normpath(d)); os.chdir(d); sys.path.insert(0, os.path.join(HERE, "sim"))
        import train as tr, score as sc
        self.tr, self.sc = tr, sc; self.build = tr.build
        self.orig_sd = torch.load("model_orig0.pt", map_location="cpu")
        D = torch.load("data.pt"); self.D = D
        self.forget = D["eval"]["clean"][0]
        self.anchors = torch.load(os.path.join(HERE, "sim", "anchors32.pt"))
        self.ref = torch.load("ref.pt"); self.sim = True
        nv = len(D["eval"]["val"][0]) // 2      # bornes hautes : vraies images du domaine (moitié du val = non vues ; contrôle vu = retain)
        self.anchor_sets = dict(coco=self.anchors, val=D["eval"]["val"][0][:nv], retain=D["eval"]["control"][0])
    def aug(self, x, seeds):
        """Vues à 32 px : crop 0.6-1 + flip + luminosité/contraste + bruit (famille voisine des copies altérées, pas identique)."""
        B, dev = x.shape[0], x.device; out = torch.empty_like(x); S = x.shape[-1]
        for i in range(B):
            g = torch.Generator().manual_seed(int(seeds[i])); r = torch.rand(8, generator=g).tolist()
            k = max(8, int(round(S * math.sqrt(0.6 + 0.4 * r[0])))); oy, ox = int(r[1] * (S - k + 1)), int(r[2] * (S - k + 1))
            im = F.interpolate(x[i:i + 1, :, oy:oy + k, ox:ox + k], size=(S, S), mode="bilinear", align_corners=False)[0]
            if r[3] < 0.5: im = im.flip(2)
            m = im.mean(); im = ((im - m) * (1 + (2 * r[4] - 1) * 0.15) + m) * (1 + (2 * r[5] - 1) * 0.15)
            out[i] = im + 0.02 * r[6] * torch.randn(im.shape, generator=g).to(dev)
        return out.clamp(0, 1)
    def inp(self, x01):
        R = self.tr.RES
        if x01.shape[-1] != R: x01 = F.interpolate(x01, size=(R, R), mode="bilinear", align_corners=False)
        return (x01 - 0.5) / 0.5
    def eval_logits(self, m):
        out = {}
        for name, (xe, _) in self.D["eval"].items():
            if len(xe): out[name] = logits(self, m, xe)
        return out


def get_env(name, dev):
    return RealEnv(dev) if name == "real" else SimEnv(name, dev)


@torch.no_grad()
def logits(env, m, xu8, idx=None, view=0, bs=250):
    """Logits comme le serveur (eval, fp16). view > 0 : vue altérée déterministe n° view."""
    m.eval(); out = []
    for i in range(0, len(xu8), bs):
        n = len(xu8[i:i + bs]); ids = idx[i:i + bs] if idx is not None else list(range(i, i + n))
        x = env.views(xu8[i:i + bs], ids, [view] * n)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(m(env.inp(x)).float().cpu())
    return torch.cat(out)


# ----------------------------------------------------------------------------- réseau : tronc gelé partagé + k derniers blocs entraînés
def trunk(m, x, k):
    x = m.patch_embed(x); x = m._pos_embed(x); x = m.patch_drop(x); x = m.norm_pre(x)
    for b in m.blocks[:len(m.blocks) - k]: x = b(x)
    return x


def top(m, h, k):
    for b in m.blocks[len(m.blocks) - k:]: h = b(h)
    return m.forward_head(m.norm(h))


class Net:
    def __init__(self, env, k):
        self.env, self.k = env, k
        self.teacher, self.student = env.new_model(), env.new_model()
        for p in list(self.teacher.parameters()) + list(self.student.parameters()): p.requires_grad_(False)
        s = self.student; k = min(k, len(s.blocks)); self.k = k
        mods = list(s.blocks[len(s.blocks) - k:]) + [s.norm, s.fc_norm, s.head]
        self.params = [p for mod in mods for p in mod.parameters()]
        if k == len(s.blocks):
            self.params += [p for n, p in s.named_parameters() if n.startswith(("patch_embed", "pos_embed", "cls_token"))]
        sub = os.environ.get("K3_TRAIN", "")            # sous-ensemble de paramètres : attn | mlp | ls | nohead | first:K
        if sub:
            named = dict(s.named_parameters()); ids = {id(p) for p in self.params}
            def keep(n):
                if sub == "attn": return ".attn." in n
                if sub == "mlp": return ".mlp." in n
                if sub == "ls": return ".ls1." in n or ".ls2." in n
                if sub == "nohead": return not n.startswith(("head", "norm", "fc_norm"))
                if sub.startswith("first:"): return n.startswith(("patch_embed", "pos_embed", "cls_token")) or (n.startswith("blocks.") and int(n.split(".")[1]) < int(sub[6:]))
                return True
            self.params = [p for n, p in named.items() if id(p) in ids and keep(n)]
        for p in self.params: p.requires_grad_(True)

    def forward(self, x01):
        xb = self.env.inp(x01)
        with torch.autocast("cuda", dtype=AMP):
            if self.k == len(self.student.blocks):
                with torch.no_grad(): zt = self.teacher(xb).float()
                zs = self.student(xb).float()
            else:
                with torch.no_grad():
                    h = trunk(self.teacher, xb, self.k); zt = top(self.teacher, h, self.k).float()
                zs = top(self.student, h, self.k).float()
        return zt, zs

    def interp(self, alpha):
        """élève <- orig + alpha * (élève - orig)."""
        with torch.no_grad():
            for (n, p), p0 in zip(self.student.named_parameters(), self.teacher.parameters()): p.copy_(p0 + alpha * (p - p0))


def cosine(step, total, warm=10):
    return (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def adv_negatives(env, teacher, x01, ycls, steps=4, eps=8 / 255, T_adv=1.0):
    """Négatifs « collision de features » : on pousse des images (ancres, vues fortes) vers une classe du jeu à oublier par quelques pas de
    PGD ciblé sur orig. Orig y devient sûr de la classe sans que ce soit une image à oublier : l'élève doit y rester égal à orig."""
    x0 = x01.detach(); x = x0.clone(); a = 2.5 * eps / steps
    for _ in range(steps):
        x.requires_grad_(True)
        with torch.autocast("cuda", dtype=AMP):
            z = teacher(env.inp(x)).float()
        loss = F.cross_entropy(z / T_adv, ycls)
        g, = torch.autograd.grad(loss, x)
        x = (x.detach() - a * g.sign()).clamp(x0 - eps, x0 + eps).clamp(0, 1)
    return x.detach()


def anchor_batch(env, B, forget_pool=None, mosaic_frac=0.0, src="coco"):
    names = src.split("+"); A = env.anchor_sets[names[torch.randint(0, len(names), (1,)).item()]]
    n_mos = int(round(B * mosaic_frac)) if forget_pool is not None else 0
    u = env.to01(A[torch.randint(0, len(A), (B - n_mos,))])
    if n_mos:
        idx = torch.randint(0, len(forget_pool), (n_mos * 4,)); S = u.shape[-1]
        x = F.interpolate(env.to01(env.forget[forget_pool[idx]]), size=(S // 2, S // 2), mode="bilinear", align_corners=False).view(n_mos, 4, 3, S // 2, S // 2)
        u = torch.cat([u, torch.cat([torch.cat([x[:, 0], x[:, 1]], 3), torch.cat([x[:, 2], x[:, 3]], 3)], 2)])
    return torch.where((torch.rand(len(u), device=env.dev) < 0.5)[:, None, None, None], u.flip(3), u)


def rand_views(n, clean_frac=0.5):
    v = torch.randint(1, 10 ** 6, (n,)); v[torch.rand(n) < clean_frac] = 0
    return v.tolist()


# ----------------------------------------------------------------------------- fold : sorties hors-pli
def displaced(zt, y, kappa, mode):
    """Cible de déplacement pour les images du pli : la classe mémorisée est ramenée kappa logits SOUS la meilleure autre classe."""
    if mode == "uniform": return torch.zeros_like(zt)
    z = zt.clone(); ar = torch.arange(len(y), device=z.device)
    other = z.scatter(1, y[:, None], -1e4).max(1).values
    z[ar, y] = torch.minimum(z[ar, y], other - kappa)
    return z


def run_fold(args, env):
    net = Net(env, args.blocks); X = env.forget; N = len(X); dev = env.dev
    z0 = logits(env, net.teacher, X); yhat = z0.argmax(1); yd = yhat.to(dev)
    g = torch.Generator().manual_seed(1000 + args.rep)
    if args.by_class:                       # plis par classes entières (aucune image de la classe ne reste dans le « retain »)
        cls = yhat.unique(); cf = torch.randperm(len(cls), generator=g) % args.k
        fold_of = torch.zeros(N, dtype=torch.long)
        for c, f in zip(cls.tolist(), cf.tolist()): fold_of[yhat == c] = f
    else:
        fold_of = torch.randperm(N, generator=g) % args.k
    f_idx = (fold_of == args.fold).nonzero()[:, 0]; r_idx = (fold_of != args.fold).nonzero()[:, 0]
    torch.manual_seed(args.seed + 17 * args.fold + 1000 * args.rep)
    opt = torch.optim.AdamW(net.params, lr=args.lr, weight_decay=0.0)
    nb = math.ceil(len(f_idx) / args.bf)
    total = nb * args.e1 + args.relax
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total) if args.cos else min(1.0, (s + 1) / 10))
    t0 = time.time(); step = 0

    def retain_terms():
        xs = []; n_r = 0; ridx = None
        if args.br and len(r_idx):
            ridx = r_idx[torch.randint(0, len(r_idx), (args.br,))]
            xs.append(env.views(X[ridx], ridx.tolist(), rand_views(len(ridx)))); n_r = len(ridx)
        if args.ba: xs.append(anchor_batch(env, args.ba, r_idx, args.mosaic))
        return xs, n_r, ridx

    def retain_loss(zt, zs, n_r, ridx):
        loss = 0.0
        if n_r:
            loss = loss + args.w_r * kl_t(zt[:n_r], zs[:n_r]).mean() * T * T + args.gamma * F.cross_entropy(zs[:n_r], yd[ridx.to(dev)])
        if args.ba: loss = loss + args.lam * kl_t(zt[n_r:], zs[n_r:]).mean() * T * T
        return loss

    for ep in range(args.e1):               # ---- déplacement du pli + maintien du reste
        perm = f_idx[torch.randperm(len(f_idx))]
        for i in range(0, len(perm), args.bf):
            idx = perm[i:i + args.bf]
            xf = env.views(X[idx], idx.tolist(), rand_views(len(idx)))
            xs, n_r, ridx = retain_terms()
            zt, zs = net.forward(torch.cat([xf] + xs)); nf = len(idx)
            if args.disp == "ascent":
                ce = F.cross_entropy(zs[:nf].detach(), yd[idx.to(dev)], reduction="none"); w = (ce < args.cap).float()
                lf = -(w * kl_t(zt[:nf], zs[:nf])).sum() / w.sum().clamp_min(1) * T * T
            else:
                lf = kl_t(displaced(zt[:nf], yd[idx.to(dev)], args.kappa, args.disp), zs[:nf]).mean() * T * T
            loss = args.w_f * lf + retain_loss(zt[nf:], zs[nf:], n_r, ridx)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.params, 1.0); opt.step(); sched.step(); step += 1
    snaps = {}
    def snap(tag):
        V = args.views
        o = dict(clean=logits(env, net.student, X[f_idx], f_idx.tolist(), 0),
                 views=torch.stack([logits(env, net.student, X[f_idx], f_idx.tolist(), v) for v in range(1, V + 1)]) if V else None)
        if env.sim: o["altered"] = logits(env, net.student, env.D["eval"]["altered"][0][f_idx])
        ce = F.cross_entropy(o["clean"], yhat[f_idx], reduction="none")
        print(f"  [{tag}] pli propre acc {(o['clean'].argmax(1) == yhat[f_idx]).float().mean():.3f} CE {ce.mean():.3f}±{ce.std():.3f} ({time.time() - t0:.0f}s)", flush=True)
        snaps[tag] = o
    snap("r0")
    marks = sorted(set(int(v) for v in args.snaps.split(",") if v)) if args.snaps else []
    for s in range(1, args.relax + 1):       # ---- relâchement : seulement le reste (les sorties du pli reviennent vers la généralisation)
        xs, n_r, ridx = retain_terms()
        zt, zs = net.forward(torch.cat(xs)); loss = retain_loss(zt, zs, n_r, ridx)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.params, 1.0); opt.step(); sched.step()
        if s in marks or s == args.relax: snap(f"r{s}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(idx=f_idx, snaps=snaps, args=vars(args)), args.out)
    print(f"OK {args.out} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- merge + évaluation des cibles
def run_merge(args, env):
    files = sorted(sum((glob.glob(p) for p in args.inputs.split(",")), []))
    assert files, "aucun fichier"
    N = len(env.forget); tags = list(torch.load(files[0])["snaps"].keys())
    for tag in tags:
        acc = {}; cnt = torch.zeros(N)
        for f in files:
            d = torch.load(f); idx = d["idx"]; o = d["snaps"][tag]; cnt[idx] += 1
            for key in ("clean", "views", "altered"):
                if o.get(key) is None: continue
                p = F.softmax(o[key].float() / T, -1)
                if key not in acc: acc[key] = torch.zeros((p.shape[0], N, 100) if key == "views" else (N, 100))
                if key == "views": acc[key][:, idx] += p
                else: acc[key][idx] += p
        assert cnt.min() > 0, f"images sans cible : {(cnt == 0).sum()}"
        tgt = {k: v / (cnt[None, :, None] if k == "views" else cnt[:, None]) for k, v in acc.items()}
        out = args.out.replace(".pt", f"_{tag}.pt"); os.makedirs(os.path.dirname(out) or ".", exist_ok=True); torch.save(tgt, out)
        print(f"{tag}: {len(files)} fichiers, {cnt.mean():.1f} passes/image -> {out}", flush=True)
        if env.sim: eval_targets(env, tgt, tag)


def p2z(p): return T * torch.log(p.clamp_min(1e-8))


def transform(z, y, s=1.0, b=0.0):
    """Réglages globaux : b = décalage du logit de la classe d'origine, s = échelle des logits (s<1 adoucit)."""
    z = z.clone(); z[torch.arange(len(y)), y] += b
    return z * s


def eval_targets(env, tgt, tag="", grid=True):
    """Simulateur : note les cibles elles-mêmes (candidat = cibles sur forget, orig ailleurs)."""
    ref = env.ref; sc = env.sc; y = ref["orig"]["clean"].argmax(1)
    zc = p2z(tgt["clean"]); za = p2z(tgt["altered"]) if "altered" in tgt else zc
    base = dict(control=ref["orig"]["control"], control_unseen=ref["orig"]["control_unseen"], val=ref["orig"]["val"])
    def sco(s, b, alt):
        return sc.score(dict(base, clean=transform(zc, y, s, b), altered=transform(alt, y, s, b)), ref, verbose=False)
    o = sco(1, 0, za); c, a = o["clean"], o["altered"]
    print(f"  cibles {tag}: MS {c['ms']:.3f}/{a['ms']:.3f} PS {c['ps']:.3f}/{a['ps']:.3f} forget {o['forget']:.4f} | acc {c['acc']:.3f} CE {c['ce_mean']:.2f}±{c['ce_std']:.2f} (gold {c['gold_mean']:.2f}±{c['gold_std']:.2f})")
    if grid:
        best = max(((sco(s, b, za)["forget"], s, b) for s in (0.5, 0.7, 0.85, 1, 1.2, 1.5, 2) for b in (-4, -3, -2, -1, 0, 1, 2, 3, 4)), key=lambda t: t[0])
        o2 = sco(best[1], best[2], za); bp = max(((sco(s, b, za)["ps"], s, b) for s in (0.5, 0.7, 0.85, 1, 1.2, 1.5, 2) for b in (-4, -2, -1, 0, 1, 2, 4)), key=lambda t: t[0])
        o3 = sco(best[1], best[2], zc)
        print(f"    meilleur réglage (s={best[1]}, b={best[2]}): forget {best[0]:.4f} (MS {o2['ms']:.3f} PS {o2['ps']:.3f}) | PS max {bp[0]:.3f} (s={bp[1]}, b={bp[2]}) | même réglage, cible propre réutilisée pour l'altéré : forget {o3['forget']:.4f}")
    return o


def run_evalt(args, env):
    eval_targets(env, torch.load(args.targets), os.path.basename(args.targets))


BULK = None      # (k, d) : les autres classes au-delà des k meilleures sont abaissées de d logits (masse concentrée sur les concurrentes proches)


def cap_targets(zt, y, mhat, tb, obias=None):
    """Cible par vue : logits d'orig sur la vue ; si orig y est plus sûr que la marge visée mhat(x), on plafonne la marge de la classe
    d'origine à mhat et on resserre les autres classes (échelle tb). Sinon (orig déjà « non vu ») la cible est orig lui-même."""
    zt = zt.float(); ar = torch.arange(len(y), device=zt.device)
    o = zt.clone(); o[ar, y] = -1e4; mx = o.max(1).values
    capped = (zt[ar, y] - mx) > mhat
    tc = mx[:, None] + (zt - mx[:, None]) * (tb[:, None] if torch.is_tensor(tb) else tb)
    if BULK is not None:
        k, d = BULK; kth = o.topk(k, 1).values[:, -1:]; low = (zt < kth); low[ar, y] = False
        tc = tc - d * low.float()
    if obias is not None:               # biais « voisins » sur les autres classes ; la marge se mesure après
        tc = tc + obias; o2 = tc.clone(); o2[ar, y] = -1e4; mx = o2.max(1).values
    tc[ar, y] = mx + mhat
    return torch.where(capped[:, None], tc, zt), capped


def logit_mse(t, s):
    t = t - t.mean(1, keepdim=True); s = s.float() - s.float().mean(1, keepdim=True)
    return ((t - s) ** 2).mean()


# ----------------------------------------------------------------------------- distillation finale
def run_distill(args, env):
    net = Net(env, args.blocks); X = env.forget; N = len(X); dev = env.dev
    z0 = logits(env, net.teacher, X); yhat = z0.argmax(1)
    tgt = torch.load(args.targets)
    zc = transform(p2z(tgt["clean"]), yhat, args.s, args.b)
    zv = torch.stack([transform(p2z(v), yhat, args.s, args.b) for v in tgt["views"]]) if tgt.get("views") is not None else None
    V = 0 if zv is None else zv.shape[0]
    cap = args.cap_mode and "mhat" in tgt
    yd = yhat.to(dev)
    if cap:
        mhat = (tgt["mhat"] + args.b).to(dev); tb = tgt["tb"]; obias = tgt.get("obias")
        obias = obias.to(dev) if obias is not None else None
        tbv = tgt.get("tbv"); tbv = tbv.to(dev) if tbv is not None else None
        global BULK
        if tgt.get("bulk"): BULK = tgt["bulk"]; print("bulk", BULK, flush=True)
    torch.manual_seed(args.seed)
    if args.llrd < 1.0:                      # pas d'apprentissage décroissant vers les premières couches (elles portent les features générales)
        nbk = len(net.student.blocks); groups = {}; ids = {id(p) for p in net.params}
        for n_, p in net.student.named_parameters():
            if id(p) not in ids: continue
            depth = int(n_.split(".")[1]) + 1 if n_.startswith("blocks.") else (0 if n_.startswith(("patch_embed", "pos_embed", "cls_token")) else nbk + 1)
            groups.setdefault(depth, []).append(p)
        opt = torch.optim.AdamW([dict(params=ps_, lr=args.lr * args.llrd ** (nbk + 1 - d_)) for d_, ps_ in groups.items()], weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(net.params, lr=args.lr, weight_decay=0.0)
    fisher = None
    if args.ewc > 0:                         # inspiré de « certified removal » : protéger les poids là où la courbure des données à CONSERVER est forte.
        # Fisher diagonale du modèle d'origine sur des substituts des données restantes : ancres + petits crops d'images à oublier (contenu du domaine non reconnu).
        fisher = [torch.zeros_like(p) for p in net.params]; theta0 = [p.detach().clone() for p in net.params]
        for b_ in range(args.ewc_batches):
            u_ = anchor_batch(env, 12, torch.arange(N), 0.0, args.anchor_src); j_ = torch.randint(0, N, (12,))
            xs_ = env.views(X[j_], j_.tolist(), (-torch.randint(1, 10 ** 6, (12,))).tolist())
            _, zs_ = net.forward(torch.cat([u_, xs_]))
            ys_ = torch.multinomial(F.softmax(zs_.detach().float(), 1), 1).squeeze(1)       # étiquette tirée de la loi du modèle (Fisher vraie)
            gr = torch.autograd.grad(F.cross_entropy(zs_.float(), ys_), net.params, allow_unused=True)
            for f_, g_ in zip(fisher, gr):
                if g_ is not None: f_.add_(g_.detach().float() ** 2)
        tot = sum(f_.sum() for f_ in fisher); cnt_ = sum(f_.numel() for f_ in fisher)
        fisher = [(f_ / (tot / cnt_)).clamp_max(args.ewc_clip) for f_ in fisher]                # moyenne 1 ; plafonnée pour ne pas geler quelques poids
        print(f"  Fisher : {args.ewc_batches} lots, part des poids > 1 : {sum((f_ > 1).sum() for f_ in fisher) / cnt_:.3f}", flush=True)
    ema = [p.detach().clone() for p in net.params] if args.ema > 0 else None
    nb = math.ceil(N / args.bf); total = nb * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total, 20))
    A0 = env.anchors[:1000]; za0 = logits(env, net.teacher, A0)
    scaler = torch.amp.GradScaler("cuda") if AMP == torch.float16 else None
    t0 = time.time(); step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        for i in range(0, N, args.bf):
            idx = perm[i:i + args.bf]; n = len(idx)
            if cap:                          # cible calculée sur chaque vue à partir d'orig (plafond de marge)
                v = torch.randint(1, 10 ** 6, (n,)); r = torch.rand(n)
                v[r < args.clean_frac] = 0; v[r > 1 - args.strong_frac] *= -1; tg = None
            elif V and args.view_targets:    # cible propre à la vue (vues fixes 1..V) ou image propre
                v = torch.randint(0, V + 1, (n,)); v[torch.rand(n) < args.clean_frac] = 0
                tg = torch.where((v == 0)[:, None], zc[idx], zv[(v - 1).clamp_min(0), idx])
            else:                            # même cible pour toutes les vues aléatoires de l'image
                v = torch.tensor(rand_views(n, args.clean_frac)); tg = zc[idx]
            xf = env.views(X[idx], idx.tolist(), v.tolist())
            if args.syn_frac > 0 and getattr(env, "syn", None) is not None and env.syn_of is not None:
                # une partie des exemples : une VARIANTE synthétique de l'image (traitée comme une copie altérée : même marge visée, abaissée de view_delta)
                pick = (torch.rand(n) < args.syn_frac).nonzero().flatten()
                if len(pick):
                    rows = env.syn_of[idx[pick], torch.randint(0, env.syn_of.shape[1], (len(pick),))]
                    xs_ = env.to01(env.syn[rows]); fl = torch.rand(len(pick), device=dev) < 0.5
                    xf[pick.to(dev)] = torch.where(fl[:, None, None, None], xs_.flip(3), xs_); v[pick] = 7
            u = anchor_batch(env, args.ba, torch.arange(N), args.mosaic, args.anchor_src)
            if args.mixf > 0:                # négatifs « contenu du domaine dilué » : mélange image à oublier x ancre, la cible y reste orig
                nm = int(round(args.ba * args.mixf)); j = torch.randint(0, N, (nm,))
                lamb = (0.3 + 0.4 * torch.rand(nm, 1, 1, 1, device=dev)); xfj = env.views(X[j], j.tolist(), rand_views(nm))
                if args.mix_mode == "cut":   # moitié gauche/droite ou haut/bas
                    S_ = u.shape[-1]; mk = torch.zeros(nm, 1, S_, S_, device=dev); h = S_ // 2
                    for q in range(nm):
                        r = int(torch.randint(0, 4, (1,)))
                        if r == 0: mk[q, :, :, :h] = 1
                        elif r == 1: mk[q, :, :, h:] = 1
                        elif r == 2: mk[q, :, :h] = 1
                        else: mk[q, :, h:] = 1
                    mixed = mk * xfj + (1 - mk) * u[-nm:]
                else: mixed = lamb * xfj + (1 - lamb) * u[-nm:]
                u = torch.cat([u[:-nm], mixed])
            if args.adv_frac > 0:            # une partie des ancres (et des vues fortes d'images à oublier) poussée vers une classe du jeu à oublier
                na = int(round(args.ba * args.adv_frac)); j = torch.randint(0, N, (na,))
                base = u[:na] if args.adv_base == "anchor" else env.views(X[j], j.tolist(), (-torch.randint(1, 10 ** 6, (na,))).tolist())
                ycl = (torch.randint(0, zc.shape[1], (na,), device=dev) if args.adv_cls == "all" else
                       yd[torch.randint(0, N, (na,), device=dev)] if args.adv_cls == "rand" else yd[j.to(dev)])
                u = torch.cat([adv_negatives(env, net.teacher, base, ycl, args.adv_steps, args.adv_eps / 255), u[na:]])
            zt, zs = net.forward(torch.cat([xf, u]))
            if cap:
                mh = mhat[idx.to(dev)] - args.view_delta * (v != 0).float().to(dev)      # vues altérées : marge visée plus basse (l'élève y sous-oublie sinon)
                src = zt[:n]
                if args.others_src == "clean":   # mêmes autres classes (donc même mauvaise classe) pour toutes les copies : logits d'orig sur l'image PROPRE,
                    zc0 = z0[idx].to(dev)        # sauf si orig ne reconnaît déjà plus la vue (alors on garde ses logits sur la vue)
                    ar_ = torch.arange(n, device=dev); o_ = src.float().clone(); o_[ar_, yd[idx.to(dev)]] = -1e4
                    known = (src.float()[ar_, yd[idx.to(dev)]] - o_.max(1).values) > mh
                    src = torch.where(known[:, None], zc0, src.float())
                tg, _ = cap_targets(src, yd[idx.to(dev)], mh, tbv[idx.to(dev)] if tbv is not None else tb, obias[idx.to(dev)] if obias is not None else None)
                if args.strong_mode == "keep":   # vues fortes = négatifs du domaine : la cible y est orig, sans plafond
                    tg = torch.where((v < 0).to(dev)[:, None], zt[:n].float(), tg)
            tg = tg.to(dev)
            lf = args.w_kl * kl_t(tg, zs[:n]).mean() * T * T + (args.w_mse * logit_mse(tg, zs[:n]) if args.w_mse else 0.0)
            if args.w_ce:                    # cale directement la perte par image (statistique du mean/std score)
                yb = yd[idx.to(dev)] if cap else yhat[idx].to(dev)
                lf = lf + args.w_ce * F.smooth_l1_loss(F.cross_entropy(zs[:n].float(), yb, reduction="none"), F.cross_entropy(tg.float(), yb, reduction="none"))
            la = kl_t(zt[n:], zs[n:]).mean() * T * T + (args.w_mse_a * logit_mse(zt[n:].float(), zs[n:]) if args.w_mse_a else 0.0)
            loss = lf + args.lam * la
            if fisher is not None:
                loss = loss + args.ewc * sum((f_ * (p_ - t_) ** 2).sum() for f_, p_, t_ in zip(fisher, net.params, theta0))
            opt.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.params, 1.0); opt.step()
            else:
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.params, 1.0); scaler.step(opt); scaler.update()
            sched.step(); step += 1
            if ema is not None:
                with torch.no_grad():
                    for e_, p_ in zip(ema, net.params): e_.mul_(args.ema).add_(p_.detach(), alpha=1 - args.ema)
            if step in (20, 60): print(f"  pas {step}/{total} ({time.time() - t0:.0f}s)", flush=True)
            if step % 200 == 0: print(f"  pas {step}/{total} L_F {float(lf):.4f} L_A {float(la):.4f} ({time.time() - t0:.0f}s)", flush=True)
    if ema is not None:                      # on évalue et on sauve la moyenne glissante des poids
        with torch.no_grad():
            for e_, p_ in zip(ema, net.params): p_.copy_(e_)
    if args.repair > 0:                      # phase de réparation : ancres (+ vues fortes gardées à orig) seulement, les images à oublier ne sont plus poussées
        opt2 = torch.optim.AdamW(net.params, lr=args.repair_lr, weight_decay=0.0)
        for r in range(args.repair):
            u = anchor_batch(env, args.ba, torch.arange(N), args.mosaic, args.anchor_src)
            j = torch.randint(0, N, (args.bf // 2,)); xs = env.views(X[j], j.tolist(), (-torch.randint(1, 10 ** 6, (len(j),))).tolist())
            zt, zs = net.forward(torch.cat([u, xs])); nu = len(u)
            tgs, _ = cap_targets(zt[nu:], yd[j.to(dev)], mhat[j.to(dev)] - args.view_delta, tbv[j.to(dev)] if tbv is not None else tb) if cap else (zt[nu:], None)
            loss = kl_t(zt[:nu], zs[:nu]).mean() * T * T + args.repair_wf * kl_t(tgs, zs[nu:]).mean() * T * T
            opt2.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.params, 1.0); opt2.step()
        print(f"  réparation : {args.repair} pas ({time.time() - t0:.0f}s)", flush=True)
    theta = [p.detach().clone() for p in net.student.parameters()]
    for alpha in [float(a) for a in args.alphas.split(",")]:
        with torch.no_grad():
            for p, p1, p0 in zip(net.student.parameters(), theta, net.teacher.parameters()): p.copy_(p0 + alpha * (p1 - p0))
        zs = logits(env, net.student, X); ce = F.cross_entropy(zs, yhat, reduction="none")
        zsa = logits(env, net.student, A0); kla = kl_t(za0, zsa).mean().item(); agree = (zsa.argmax(1) == za0.argmax(1)).float().mean().item()
        print(f"[alpha={alpha}] forget propre acc {(zs.argmax(1) == yhat).float().mean():.3f} CE {ce.mean():.3f}±{ce.std():.3f} KL(cible||sub) {kl_t(zc, zs).mean():.4f} "
              f"(orig {kl_t(zc, z0).mean():.4f}) | ancres accord {agree:.4f} KL {kla:.4f}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if env.sim:
            el = env.eval_logits(net.student); o = env.sc.score(el, env.ref)
            if args.save_logits: torch.save(el, args.out.replace(".pt", f"_a{alpha}_logits.pt"))
            json.dump({k: v for k, v in o.items() if not isinstance(v, dict)} | dict(args=vars(args), alpha=alpha), open(args.out.replace(".pt", f"_a{alpha}.json"), "w"))
        if args.save or not env.sim:
            path = args.out.replace(".pt", f"_a{alpha}.pt"); os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            sd = {k: v.detach().cpu().float().contiguous() for k, v in net.student.state_dict().items()}
            assert all(torch.isfinite(v).all() for v in sd.values()); torch.save(sd, path)
            env.build().load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            print(f"OK {path} ({os.path.getsize(path) / 1e6:.0f} Mo)", flush=True)


def run_avg(args, env):
    """Moyenne de poids de plusieurs élèves (même départ orig) : garde le comportement appris sur les images à oublier, moyenne la dérive propre à chaque graine."""
    files = sorted(sum((glob.glob(p) for p in args.inputs.split(",")), [])); assert len(files) >= 1, "aucun fichier"
    sds = [torch.load(f, map_location="cpu") for f in files]
    avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).float() if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    m = env.build(); m.load_state_dict(avg, strict=True); m = m.to(env.dev).eval()
    X = env.forget; z0 = logits(env, env.new_model(), X); yhat = z0.argmax(1); zs = logits(env, m, X); ce = F.cross_entropy(zs, yhat, reduction="none")
    A0 = env.anchors[:1000]; za0 = logits(env, env.new_model(), A0); zsa = logits(env, m, A0)
    print(f"moyenne de {len(files)} : forget propre acc {(zs.argmax(1) == yhat).float().mean():.3f} CE {ce.mean():.3f}±{ce.std():.3f} | ancres accord {(zsa.argmax(1) == za0.argmax(1)).float().mean():.4f} KL {kl_t(za0, zsa).mean():.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if env.sim:
        o = env.sc.score(env.eval_logits(m), env.ref)
        json.dump({k: v for k, v in o.items() if not isinstance(v, dict)} | dict(files=files), open(args.out.replace(".pt", ".json"), "w"))
    if args.save or not env.sim:
        assert all(torch.isfinite(v).all() for v in avg.values()); torch.save({k: v.contiguous() for k, v in avg.items()}, args.out)
        env.build().load_state_dict(torch.load(args.out, map_location="cpu", weights_only=True), strict=True); print(f"OK {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fold", "merge", "evalt", "distill", "avg"])
    ap.add_argument("--env", default="real"); ap.add_argument("--out", default="out.pt"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bf", type=int, default=64); ap.add_argument("--br", type=int, default=64); ap.add_argument("--ba", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--mosaic", type=float, default=0.0)
    # fold
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--by_class", type=int, default=0)
    ap.add_argument("--e1", type=int, default=3); ap.add_argument("--relax", type=int, default=100); ap.add_argument("--snaps", default="")
    ap.add_argument("--disp", default="second", choices=["second", "uniform", "ascent"]); ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--cap", type=float, default=3.0); ap.add_argument("--w_f", type=float, default=1.0); ap.add_argument("--w_r", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0); ap.add_argument("--cos", type=int, default=0); ap.add_argument("--views", type=int, default=4)
    # merge / evalt / distill
    ap.add_argument("--inputs", default=""); ap.add_argument("--targets", default="")
    ap.add_argument("--s", type=float, default=1.0); ap.add_argument("--b", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--alphas", default="1.0"); ap.add_argument("--clean_frac", type=float, default=0.5)
    ap.add_argument("--view_targets", type=int, default=1); ap.add_argument("--save", type=int, default=0)
    ap.add_argument("--cap_mode", type=int, default=1); ap.add_argument("--strong_frac", type=float, default=0.0)
    ap.add_argument("--anchor_src", default="coco"); ap.add_argument("--w_ce", type=float, default=0.0); ap.add_argument("--save_logits", type=int, default=0)
    ap.add_argument("--adv_frac", type=float, default=0.0); ap.add_argument("--adv_steps", type=int, default=4); ap.add_argument("--adv_eps", type=float, default=8.0)
    ap.add_argument("--adv_base", default="anchor", choices=["anchor", "strong"]); ap.add_argument("--adv_cls", default="rand", choices=["rand", "same", "all"])
    ap.add_argument("--strong_mode", default="cap", choices=["cap", "keep"])
    ap.add_argument("--view_delta", type=float, default=0.0)
    ap.add_argument("--syn_frac", type=float, default=0.0)
    ap.add_argument("--ewc", type=float, default=0.0); ap.add_argument("--ewc_batches", type=int, default=150); ap.add_argument("--ewc_clip", type=float, default=100.0)
    ap.add_argument("--others_src", default="view", choices=["view", "clean"])
    ap.add_argument("--llrd", type=float, default=1.0); ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--repair", type=int, default=0); ap.add_argument("--repair_lr", type=float, default=3e-5); ap.add_argument("--repair_wf", type=float, default=0.0)
    ap.add_argument("--mixf", type=float, default=0.0); ap.add_argument("--mix_mode", default="mix", choices=["mix", "cut"])
    ap.add_argument("--w_kl", type=float, default=1.0); ap.add_argument("--w_mse", type=float, default=0.0); ap.add_argument("--w_mse_a", type=float, default=0.0)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out) if args.env == "real" else args.out
    env = get_env(args.env, "cuda")
    print(f"env={env.name} {vars(args)}", flush=True)
    dict(fold=run_fold, merge=run_merge, evalt=run_evalt, distill=run_distill, avg=run_avg)[args.mode](args, env)


if __name__ == "__main__":
    main()
