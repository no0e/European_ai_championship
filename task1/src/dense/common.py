"""Tâche 1, détecteur dense — briques communes (chemins, réserve d'images, masques, modèle).

Constat qui fonde la méthode (mesuré sur le train, voir README) : dans les images « full » ET dans les images
« objet transplanté », le tatouage est à l'échelle de l'image entière ; l'image est ensuite masquée :
    y = M · embed(x, m) + (1 − M) · x
(M grand avec des trous pour full, M petit en forme d'objet pour la transplantation). Seuls les rectangles
(outpainting) portent le tatouage à une autre échelle. On peut donc fabriquer autant d'exemples exacts que l'on
veut avec les tatoueurs officiels, et entraîner un détecteur dense « tatoué / pas tatoué » par méthode.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent


def _find_task_root() -> Path:
    for p in [HERE, *HERE.parents]:
        if (p / "starter_kit" / "wmloc").exists():
            return p
    raise SystemExit("starter_kit introuvable au-dessus de " + str(HERE))


ROOT = _find_task_root()
KIT = ROOT / "starter_kit"
DATA = Path(os.environ.get("T1_DATA", ROOT))                     # release/, clean/DIV2K_valid_HR/, sam/out/
MODELS = Path(os.environ.get("T1_MODELS", KIT / "models"))       # official watermark models (download_models.py)
WORK = Path(os.environ.get("T1_WORK", HERE))                     # pairs/, runs/, cache/, out/
RELEASE = DATA / "release"
SAM_OUT = DATA / "sam" / "out"
sys.path.insert(0, str(KIT))
for _p in (ROOT, ROOT / "paulin"):
    if (_p / "t1_metric.py").exists():
        sys.path.insert(0, str(_p)); break
METHODS = ("pixelseal", "trustmark", "stegastamp")


# ------------------------------------------------------------------ données réelles
def load_meta(split):
    return json.loads((RELEASE / split / "metadata.json").read_text())


def load_img(split, name) -> np.ndarray:
    return np.asarray(Image.open(RELEASE / split / "images" / f"{name}.png").convert("RGB"))


def load_gt128() -> dict:
    z = np.load(RELEASE / "train" / "masks_128.npz")
    return dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def global_ll(split) -> dict:
    s = np.load(RELEASE / f"signals_{split}.npz")
    return dict(zip(s["name"].tolist(), s["global_ll"].tolist()))


def best_split(score, y):
    s = np.sort(np.unique(score))
    c = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    acc = [((score > t) == y).mean() for t in c]
    return float(c[int(np.argmax(acc))])


def regime_thresholds() -> dict:
    """Seuil par méthode sur la log-vraisemblance globale du message (appris sur le train) : > seuil = full."""
    meta, ll = load_meta("train"), global_ll("train")
    out = {}
    for m in METHODS:
        sel = [x for x in meta if x["method"] == m]
        out[m] = best_split(np.array([ll[x["image_name"]] for x in sel]), np.array([x["full_watermark"] for x in sel]))
    return out


def is_rect(mask: np.ndarray) -> bool:
    ys, xs = np.nonzero(mask)
    return bool(len(ys)) and bool(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all())


def mask_kind(meta_row, m128) -> str:
    if meta_row["full_watermark"]:
        return "full"
    return "rect" if is_rect(m128) else "objet"


def ds128(m512: np.ndarray) -> np.ndarray:
    """Règle officielle 512 -> 128 : cellule = 1 si au moins 50 % de ses pixels sont tatoués."""
    lead = m512.shape[:-2]
    return m512.astype(np.float32).reshape(*lead, 128, 4, 128, 4).mean(axis=(-3, -1)) >= 0.5


# ------------------------------------------------------------------ masques synthétiques
def blob_mask(rng, frac, scale=None) -> np.ndarray:
    """Tache(s) aléatoire(s) couvrant environ `frac` de l'image (champ gaussien lissé, seuillé au quantile)."""
    s = int(scale or rng.choice([3, 4, 6, 8, 12, 16]))
    f = torch.from_numpy(rng.standard_normal((1, 1, s, s)).astype(np.float32))
    f = F.interpolate(f, size=(512, 512), mode="bicubic", align_corners=False)[0, 0].numpy()
    return f > np.quantile(f, 1 - frac)


def _place(b, rng) -> np.ndarray:
    """Place une imagette booleenne h x w au hasard dans un masque 512x512."""
    h, w = b.shape
    m = np.zeros((512, 512), bool)
    y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
    m[y0:y0 + h, x0:x0 + w] = b
    return m


def _box(rng, frac, lo=6) -> tuple:
    """(h, w) d'une boite d'aire frac*512^2, de rapport d'aspect aleatoire."""
    area = float(np.clip(frac, 1e-5, 0.9)) * 512 * 512
    ar = np.exp(rng.uniform(-1.0, 1.0))
    w = int(np.clip(round(np.sqrt(area * ar)), lo, 512))
    return int(np.clip(round(area / w), lo, 512)), w


def blob_markbench(rng, frac, dilate=2) -> np.ndarray:
    """Tache irreguliere « a la markbench » : bruit uniforme sur une grille de 4 a 16 par cote, agrandi en
    bilineaire a la taille d'une boite, seuille au quantile, puis dilate (Yao et al., arXiv:2609.16832 ;
    recette lue dans le depot garandorinc/markbench, lien avec les donnees de l'epreuve non etabli)."""
    from scipy.ndimage import binary_dilation
    q = float(rng.uniform(0.3, 0.7))                     # part de la boite reellement couverte
    h, w = _box(rng, frac / q)
    s = int(rng.integers(4, 17))
    f = torch.from_numpy(rng.random((1, 1, s, s)).astype(np.float32))
    f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    b = f > np.quantile(f, 1 - q)
    if dilate:
        b = binary_dilation(b, iterations=int(dilate))
    return _place(b, rng)


def rounded_rect(rng, frac) -> np.ndarray:
    """Rectangle a coins arrondis couvrant environ `frac` de l'image."""
    h, w = _box(rng, frac)
    r = float(rng.uniform(0.1, 0.45)) * min(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = np.maximum(0.0, np.maximum(r - 0.5 - yy, yy - (h - 1 - r + 0.5)))
    dx = np.maximum(0.0, np.maximum(r - 0.5 - xx, xx - (w - 1 - r + 0.5)))
    return _place(dy ** 2 + dx ** 2 <= r ** 2, rng)


def largest_cc(b) -> np.ndarray:
    """Plus grande composante connexe (un vrai trou est d'un seul tenant)."""
    from scipy import ndimage
    lab, n = ndimage.label(b)
    if n <= 1:
        return b
    s = np.bincount(lab.ravel())
    s[0] = 0
    return lab == int(np.argmax(s))


def hole_shape(rng, frac) -> np.ndarray:
    """Une tache de trou (A1a), d'un seul tenant et d'aire proche de `frac`.

    Champ gaussien lissé, markbench, rectangle arrondi ou forme simple. Les deux premiers se fragmentent en
    plusieurs morceaux : on ne garde que le plus gros, puis on corrige l'aire demandée une fois (sinon le nombre
    de trous par image et leur taille s'écartent du réel, mesuré sur le train)."""
    def draw(fr):
        u = rng.random()
        if u < 0.45:
            return largest_cc(blob_mask(rng, fr))
        if u < 0.75:
            return largest_cc(blob_markbench(rng, fr))
        if u < 0.90:
            return rounded_rect(rng, fr)
        return shape_mask(rng, fr)

    b = draw(frac)
    got = b.mean()
    if got < 0.6 * frac and got > 0:
        b2 = draw(min(0.9, frac * frac / got))
        if abs(b2.mean() - frac) < abs(got - frac):
            return b2
    return b


def shape_mask(rng, frac) -> np.ndarray:
    """Rectangle ou ellipse couvrant environ `frac` de l'image."""
    area = frac * 512 * 512
    ar = np.exp(rng.uniform(-1.2, 1.2))
    w = int(np.clip(np.sqrt(area * ar), 8, 512)); h = int(np.clip(area / w, 8, 512))
    x0 = int(rng.integers(0, 512 - w + 1)); y0 = int(rng.integers(0, 512 - h + 1))
    m = np.zeros((512, 512), bool)
    if rng.random() < 0.5:
        m[y0:y0 + h, x0:x0 + w] = True
    else:
        yy, xx = np.mgrid[0:512, 0:512]
        m = ((yy - y0 - h / 2) / (h / 2)) ** 2 + ((xx - x0 - w / 2) / (w / 2)) ** 2 <= 1
    return m


def rand_d4(m, rng):
    if rng.random() < 0.5:
        m = m[:, ::-1]
    return np.rot90(m, int(rng.integers(4))).copy()


def rand_shift(m, rng, max_shift=96):
    return np.roll(m, (int(rng.integers(-max_shift, max_shift + 1)), int(rng.integers(-max_shift, max_shift + 1))), (0, 1))


class MaskSampler:
    """Tire un masque M (512×512, True = tatoué) : trous (type full) ou zone en forme d'objet.

    Sources : vrais masques du train (formes réalistes, transformés), segments SAM de l'image de base (bords
    alignés sur le contenu), formes aléatoires. `exclude` : noms d'images dont on n'utilise pas le masque
    (validation)."""

    def __init__(self, exclude=()):
        from scipy import ndimage
        meta = load_meta("train")
        self.full, self.obj = [], []
        self.hole_counts, self.hole_fracs = [], []       # distribution reelle des trous (A1a)
        for m in meta:
            n = m["image_name"]
            if n in exclude:
                continue
            mk = np.asarray(Image.open(RELEASE / "train" / "masks" / f"{n}.png").convert("L")) > 127
            if m["full_watermark"]:
                self.full.append(mk)
                lab, k = ndimage.label(~mk)
                if k:
                    self.hole_counts.append(k)
                    self.hole_fracs += (np.bincount(lab.ravel())[1:] / (512.0 * 512.0)).tolist()
            elif not is_rect(mk):
                self.obj.append(mk)
        self._sam = {}

    def n_holes(self, rng) -> int:
        """Nombre de trous, tire dans la distribution reelle (mesure : mediane 2, 9e decile 8, max 24)."""
        if not self.hole_counts:
            return int(rng.integers(1, 5))
        return int(self.hole_counts[int(rng.integers(len(self.hole_counts)))])

    def hole_frac(self, rng) -> float:
        """Aire d'un trou, tiree dans la distribution reelle (mesure : mediane 0,0075, 1er decile 0,0005)."""
        if not self.hole_fracs:
            return float(np.exp(rng.uniform(np.log(0.004), np.log(0.25))))
        return float(self.hole_fracs[int(rng.integers(len(self.hole_fracs)))])

    def sam_segments(self, name):
        if name not in self._sam:
            p = SAM_OUT / "train" / f"{name}.npz"
            if not p.exists():
                p = SAM_OUT / "test" / f"{name}.npz"
            self._sam[name] = np.load(p)["masks"] if p.exists() else None
            if len(self._sam) > 4000:
                self._sam.pop(next(iter(self._sam)))
        return self._sam[name]

    def _sam_union(self, name, flip, rng, lo, hi):
        pk = self.sam_segments(name) if name else None
        if pk is None or len(pk) == 0:
            return None
        out = np.zeros((512, 512), bool)
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(len(pk)))
            seg = np.unpackbits(pk[j], count=512 * 512).reshape(512, 512).astype(bool)
            if flip:
                seg = seg[:, ::-1]
            if (out | seg).mean() > hi:
                continue
            out |= seg
        return out if out.mean() >= lo else None

    def sample(self, rng, base_name=None, flip=False):
        """Retourne (M, type) avec type dans {full, objet, vide, plein}."""
        u = rng.random()
        if u < 0.025:
            return np.zeros((512, 512), bool), "vide"
        if u < 0.05:
            return np.ones((512, 512), bool), "plein"
        if u < 0.53:                                             # type full : image tatouée moins des trous
            v = rng.random()                                     # A1a : 30 % vrais masques, 10 % SAM, 60 % taches
            holes = None                                         # (avant 40 / 30 / 30 ; mesuré : seuls 24 % des
            if v < 0.30 and self.full:                           #  vrais trous suivent les contours)
                holes = ~rand_shift(rand_d4(self.full[int(rng.integers(len(self.full)))], rng), rng, 48)
            elif v < 0.40:
                holes = self._sam_union(base_name, flip, rng, 0.003, 0.5)
            if holes is None:
                holes = np.zeros((512, 512), bool)
                for _ in range(self.n_holes(rng)):               # nombre et tailles tirés du réel
                    holes |= hole_shape(rng, self.hole_frac(rng))
                    if holes.mean() > 0.50:                      # réel : fraction trouée maximale 0,49
                        break
            if holes.mean() > 0.55:
                holes = blob_mask(rng, 0.2)
            return ~holes, "full"
        v = rng.random()                                         # type objet : petite zone tatouée
        m = None
        if v < 0.35 and self.obj:
            m = rand_shift(rand_d4(self.obj[int(rng.integers(len(self.obj)))], rng), rng)
        elif v < 0.70:
            m = self._sam_union(base_name, flip, rng, 0.002, 0.45)
        if m is None or m.mean() < 0.001:
            fr = float(np.exp(rng.uniform(np.log(0.003), np.log(0.4))))
            m = blob_mask(rng, fr) if rng.random() < 0.6 else shape_mask(rng, fr)
        return m, "objet"


def composite(x, w, M, rng, alpha=1.0):
    """y = x + α·M·(w − x) (uint8). Une fois sur quatre, bord adouci (fondu de quelques pixels).

    α = 1 redonne exactement y = M·w + (1 − M)·x. α ≠ 1 (A1b) change la force du tatouage ; les vraies images
    sortent des tatoueurs à leur force par défaut, d'où une plage centrée sur 1 (0,8-1,2)."""
    a = M.astype(np.float32)
    if rng.random() < 0.25:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, float(rng.uniform(0.7, 2.5)))
    a = (a * float(alpha))[..., None]
    xf = x.astype(np.float32)
    return np.clip(np.round(xf + a * (w.astype(np.float32) - xf)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ modèle
class DenseDet(nn.Module):
    """Détecteur dense tatoué / pas tatoué : squelette (décodeur du watermark, ou ConvNeXt générique) + FPN +
    branche pleine résolution, sortie 128×128 (logits)."""

    def __init__(self, arch: str, wm=None, timm_weights=None, scale2=False, stem_div=1):
        super().__init__()
        self.arch = arch
        self.scale2 = bool(scale2)
        if arch == "pixelseal":                                   # ConvNeXtV2 du détecteur PixelSeal (entrée 256)
            self.det = copy.deepcopy(wm.det)
            self.base_res = 256
            chans = [96, 192, 384, 768]
        elif arch == "trustmark":                                 # ResNet-50 du décodeur TrustMark (entrée 245)
            self.net = copy.deepcopy(wm.net)
            self.res = int(wm.resolution)
            self.base_res = self.res
            chans = [256, 512, 1024, 2048]
        elif arch.startswith("timm:"):                            # squelette générique, entrée 512
            import timm
            kw = dict(pretrained=bool(timm_weights), features_only=True)
            if timm_weights:
                kw["pretrained_cfg_overlay"] = dict(file=str(timm_weights))
            self.enc = timm.create_model(arch.split(":", 1)[1], **kw)
            chans = self.enc.feature_info.channels()
            self.base_res = 512
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            raise ValueError(arch)
        if self.scale2 and arch not in ("pixelseal", "trustmark"):
            raise ValueError("--scale2 n'a de sens que pour pixelseal / trustmark (entrée déjà 512 sinon)")
        if stem_div > 1:
            print("tige : " + ", ".join(shrink_stem(self.backbone_params_module(), stem_div)), flush=True)
        d = 128
        self.lat = nn.ModuleList(nn.Conv2d(c, d, 1) for c in chans)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU()) for _ in chans)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(48, 48, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Conv2d(d + 48, 128, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 1, 1))

    def backbone_params(self):
        return list(self.backbone_params_module().parameters())

    def head_params(self):
        ids = {id(p) for p in self.backbone_params()}
        return [p for p in self.parameters() if id(p) not in ids]

    def train(self, mode=True):
        super().train(mode)
        if self.arch == "trustmark":                              # BatchNorm du ResNet : statistiques figées
            self.net.eval()
        return self

    def backbone_params_module(self):
        return {"pixelseal": lambda: self.det, "trustmark": lambda: self.net}.get(self.arch, lambda: self.enc)()

    def _feats_at(self, x, res):
        """Les 4 cartes du squelette, l'image étant d'abord ramenée à `res`."""
        if self.arch == "pixelseal":
            h = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
            h = self.det.preprocess(h)
            cn, out = self.det.convnext, []
            for i in range(4):
                h = cn.stages[i](cn.downsample_layers[i](h)); out.append(h)
            return out
        if self.arch == "trustmark":
            n = self.net
            h = F.interpolate(x, size=(res,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            h = n.maxpool(n.relu(n.bn1(n.conv1(h))))
            out = []
            for k in ("layer1", "layer2", "layer3", "layer4"):
                h = getattr(n, k)(h); out.append(h)
            return out
        return list(self.enc((x - self.mean) / self.std))

    def feats(self, x):
        """x : B×3×512×512 dans [0, 1] -> 4 cartes, de la plus fine à la plus grossière.

        A3 variante 1 (`scale2`) : la même image passe aussi à la résolution double dans le même squelette, et
        les deux pyramides sont additionnées sur la grille fine. La carte la plus fine passe ainsi de 64×64 à
        128×128, la grille de sortie — un bord décalé d'une cellule coûte 0,09 de mIoU."""
        if not self.scale2:
            return self._feats_at(x, self.base_res)
        lo = self._feats_at(x, self.base_res)
        hi = self._feats_at(x, self.base_res * 2)
        return [h + F.interpolate(l, size=h.shape[-2:], mode="bilinear", align_corners=False)
                for h, l in zip(hi, lo)]

    def forward(self, x):
        f = self.feats(x)
        p = self.lat[-1](f[-1])
        p = self.smooth[-1](p)
        for i in range(len(f) - 2, -1, -1):
            p = self.lat[i](f[i]) + F.interpolate(p, size=f[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        p = F.interpolate(p, size=(128, 128), mode="bilinear", align_corners=False)
        return self.head(torch.cat([p, self.stem(x * 2 - 1)], 1))[:, 0]


def shrink_stem(mod, div=2):
    """A3 variante 2 : divise par `div` le pas de la première convolution du squelette (en place).

    `src/weak_signal_image.py::remove_stem_downsampling` met le pas à 1 : sur la tige d'un ConvNeXt (pas 4)
    cela multiplie par 16 la mémoire et le calcul, pas par 4. On divise donc, en ajustant le rembourrage pour
    garder une taille de sortie exacte. Les poids pré-entraînés sont conservés, mais lus à une autre échelle :
    à ne garder que si le gain synthétique est mesuré."""
    changed = []
    for name, m in mod.named_modules():
        if isinstance(m, nn.Conv2d) and max(m.stride) > 1:
            s = max(1, m.stride[0] // div)
            k = m.kernel_size[0]
            m.stride = (s, s)
            m.padding = ((k - s) // 2, (k - s) // 2)
            changed.append(f"{name} pas {s}, rembourrage {m.padding[0]}")
            break
    return changed or ["aucune convolution à pas > 1"]


def default_arch(method: str) -> str:
    return method if method in ("pixelseal", "trustmark") else "timm:convnext_tiny"


def build_model(method, arch, device, timm_weights=None, scale2=False, stem_div=1):
    wm = None
    if arch in ("pixelseal", "trustmark"):
        from wmloc.watermarks import load_watermark
        wm = load_watermark(arch, MODELS, str(device))
    return DenseDet(arch, wm, timm_weights, scale2, stem_div).to(device)
