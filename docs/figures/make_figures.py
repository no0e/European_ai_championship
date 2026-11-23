"""Figures for the showcase README files. Run: python make_figures.py <repo_root>"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import norm
from skimage import data as skdata
from skimage.transform import resize

ROOT = Path(sys.argv[1])
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
GRAY = "#c9c8c3"

plt.rcParams.update({
    "font.family": "Inter",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tint(img, mask, color, a=0.45):
    c = np.array(matplotlib.colors.to_rgb(color))
    out = img.copy()
    out[mask] = (1 - a) * out[mask] + a * c
    return out


def ellipse(h, w, cy, cx, ry, rx):
    y, x = np.mgrid[:h, :w]
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1


def down128(mask512):
    return mask512.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5


# ----------------------------------------------------------------------------------------------------------------
# Task 1
# ----------------------------------------------------------------------------------------------------------------
astro = resize(skdata.astronaut(), (512, 512), anti_aliasing=True)
cat = resize(skdata.chelsea(), (512, 512), anti_aliasing=True)
coffee = resize(skdata.coffee(), (512, 512), anti_aliasing=True)
rocket = resize(skdata.rocket(), (512, 512), anti_aliasing=True)


def t1_edits():
    H = 512
    # outpainting: the watermarked image is a rectangle inside a larger generated canvas
    m_out = np.zeros((H, H), bool)
    m_out[96:416, 140:372] = True
    canvas = resize(rocket, (H, H))
    small = resize(rocket, (320, 232), anti_aliasing=True)
    out_img = canvas * 0.55 + 0.45 * np.flipud(canvas)
    out_img[96:416, 140:372] = small
    # object removal: everything watermarked except the hole where an object was erased
    m_rem = np.ones((H, H), bool)
    hole = ellipse(H, H, 330, 360, 95, 70)
    m_rem[hole] = False
    from scipy.ndimage import gaussian_filter
    rem_img = astro.copy()
    blur = np.stack([gaussian_filter(astro[..., c], 18) for c in range(3)], -1)
    rem_img[hole] = blur[hole]
    # object addition: something unwatermarked pasted on top
    m_add = np.ones((H, H), bool)
    obj = ellipse(H, H, 150, 380, 70, 95)
    m_add[obj] = False
    add_img = coffee.copy()
    add_img[obj] = resize(cat, (H, H))[obj]
    # transplantation: a watermarked object pasted into a clean image
    m_tr = np.zeros((H, H), bool)
    tr = ellipse(H, H, 300, 210, 120, 110)
    m_tr[tr] = True
    tr_img = rocket.copy()
    tr_img[tr] = cat[tr]

    cases = [
        ("Outpainting", "image extended outwards", out_img, m_out),
        ("Object removal", "an object is erased", rem_img, m_rem),
        ("Object addition", "something new is pasted", add_img, m_add),
        ("Transplantation", "a watermarked object is moved", tr_img, m_tr),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), gridspec_kw={"hspace": 0.12, "wspace": 0.06})
    cmap = ListedColormap(["#ecebe7", BLUE])
    for j, (title, sub, img, m) in enumerate(cases):
        ax = axes[0, j]
        ax.imshow(tint(img, m, BLUE, 0.38))
        bare(ax)
        ax.set_title(f"{title}\n", fontsize=12)
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")
        ax = axes[1, j]
        ax.imshow(down128(m), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        bare(ax)
        frac = m.mean()
        ax.text(0.5, -0.08, f"{frac:.0%} watermarked", transform=ax.transAxes,
                ha="center", va="top", color=INK2, fontsize=10)
    axes[0, 0].text(-0.08, 0.5, "edited image", transform=axes[0, 0].transAxes, rotation=90, ha="center",
                    va="center", color=INK2)
    axes[1, 0].text(-0.08, 0.5, "what to predict", transform=axes[1, 0].transAxes, rotation=90, ha="center",
                    va="center", color=INK2)
    fig.text(0.125, 0.985, "Blue = still carries the watermark. Schematic examples on public-domain images.",
             color=INK2, fontsize=10)
    save(fig, "t1_edits.png")


def fake_residual(img, seed=0):
    """Illustration only: a pseudo-random pattern modulated by local texture, like a JND-weighted watermark."""
    rng = np.random.default_rng(seed)
    g = img.mean(axis=2)
    gy, gx = np.gradient(g)
    tex = np.sqrt(gx ** 2 + gy ** 2)
    tex = np.clip(tex / np.percentile(tex, 95), 0.15, 1)
    pat = rng.normal(size=(64, 64, 3))
    pat = resize(pat, img.shape, order=1)
    return pat * tex[..., None]


def t1_idea():
    x = astro
    res = fake_residual(x)
    M = np.ones((512, 512), bool)
    M[ellipse(512, 512, 330, 360, 95, 70)] = False
    res_vis = np.clip(0.5 + 0.18 * res, 0, 1)
    y = x.copy()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), gridspec_kw={"wspace": 0.35})
    panels = [
        (x, "clean image  x"),
        (res_vis, "embed(x, m) − x\n(official embedder, amplified)"),
        (np.where(M[..., None], [0.16, 0.47, 0.84], [0.93, 0.92, 0.9]), "random mask  M\n(real hole statistics)"),
        (tint(y, M, BLUE, 0.0), "training image\ny = M·embed(x, m) + (1−M)·x"),
        (np.where(down128(M)[..., None], [0.16, 0.47, 0.84], [0.93, 0.92, 0.9]), "label\n128 × 128"),
    ]
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, interpolation="nearest")
        bare(ax)
        ax.set_title(t, fontsize=10.5, loc="center", fontweight="medium")
    for i, sym in [(0, "→"), (1, "×"), (2, "→"), (3, "→")]:
        fig.text((axes[i].get_position().x1 + axes[i + 1].get_position().x0) / 2, 0.5, sym, ha="center",
                 va="center", fontsize=20, color=INK2)
    fig.text(0.125, 1.04, "Synthetic training pairs built with the official embedder",
             fontsize=12.5, fontweight="semibold")
    fig.text(0.125, 0.97, "The residual shown is an illustration (a texture-modulated noise), not the real PixelSeal signal.",
             fontsize=9.5, color=INK2)
    save(fig, "t1_idea.png")


def t1_metric():
    n = 16
    gt = np.zeros((n, n), bool)
    gt[3:11, 4:12] = True
    pred = np.zeros((n, n), bool)
    pred[4:12, 5:13] = True
    tp = gt & pred
    fp = pred & ~gt
    fn = gt & ~pred
    iou1 = tp.sum() / (gt | pred).sum()
    iou0 = (~gt & ~pred).sum() / (~gt | ~pred).sum()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"wspace": 0.22})
    cm = ListedColormap(["#ecebe7", BLUE])
    for ax, m, t in [(axes[0], gt, "ground truth"), (axes[1], pred, "our prediction")]:
        ax.imshow(m, cmap=cm, vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURF, linewidth=1)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, loc="center", fontweight="medium")
    comp = np.zeros((n, n, 3)) + matplotlib.colors.to_rgb("#ecebe7")
    comp[tp] = matplotlib.colors.to_rgb(BLUE)
    comp[fp] = matplotlib.colors.to_rgb(ORANGE)
    comp[fn] = matplotlib.colors.to_rgb(YELLOW)
    ax = axes[2]
    ax.imshow(comp)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=1)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("overlap", loc="center", fontweight="medium")
    for k, (c, lab) in enumerate([(BLUE, "both say watermarked"), (ORANGE, "false alarm"), (YELLOW, "missed")]):
        ax.add_patch(Rectangle((n + 0.8, 1 + 2 * k - 0.4), 0.9, 0.9, color=c, clip_on=False))
        ax.text(n + 2.1, 1 + 2 * k, lab, va="center", fontsize=10, color=INK2)
    txt = (f"IoU(watermarked)    = {tp.sum()} / {(gt | pred).sum()} = {iou1:.2f}\n"
           f"IoU(clean)          = {(~gt & ~pred).sum()} / {(~gt | ~pred).sum()} = {iou0:.2f}\n"
           f"score of this image = ({iou1:.2f} + {iou0:.2f}) / 2 = {(iou1 + iou0) / 2:.2f}")
    ax.text(n + 0.8, 9, txt, va="top", fontsize=10.5, family="DejaVu Sans Mono", color=INK)
    fig.text(0.125, 1.0, "Mean IoU of the two classes on one image (16 × 16 grid for readability; the task uses 128 × 128)",
             fontsize=12.5, fontweight="semibold")
    save(fig, "t1_metric.png")


def t1_frame():
    H = 512
    s = 1.12
    F = int(H * s)
    big = resize(astro, (F, F), anti_aliasing=True)
    off = (F - H) // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"wspace": 0.35, "width_ratios": [1, 1, 1.15]})
    ax = axes[0]
    ax.imshow(astro)
    ax.add_patch(Rectangle((0, 0), H - 1, H - 1, fill=False, ec=ORANGE, lw=3))
    bare(ax)
    ax.set_title("Watermark computed\non the image itself", loc="center", fontsize=10.5,
                 fontweight="medium")
    ax = axes[1]
    ax.imshow(big)
    ax.add_patch(Rectangle((-0.5, -0.5), F, F, fill=False, ec=BLUE, lw=3))
    ax.add_patch(Rectangle((off, off), H, H, fill=False, ec=INK, lw=1.5, ls="--"))
    bare(ax)
    ax.set_title(f"Watermark computed on a larger\nframe (≈ {s:.2f}×), then cropped",
                 loc="center", fontsize=10.5, fontweight="medium")
    ax = axes[2]
    xs = [0, 1]
    vals_ps = [0.647, 0.667]
    vals_tm = [0.634, 0.657]
    w = 0.34
    ax.bar(np.array(xs) - w / 2 - 0.01, vals_ps, w, color=BLUE, label="PixelSeal, full regime")
    ax.bar(np.array(xs) + w / 2 + 0.01, vals_tm, w, color=AQUA, label="TrustMark, full regime")
    for x0, v in zip(np.array(xs) - w / 2 - 0.01, vals_ps):
        ax.text(x0, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    for x0, v in zip(np.array(xs) + w / 2 + 0.01, vals_tm):
        ax.text(x0, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xticks(xs, ["previous ensemble\n(tuned on real images)", "1 model with frame\n(no real image)"],
                  fontsize=9.5)
    ax.set_ylim(0.6, 0.685)
    ax.set_ylabel("cross-validated mIoU")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Cross-validated mIoU", loc="left")
    fig.text(0.125, 1.03, "Watermarked frame larger than the image",
             fontsize=12.5, fontweight="semibold")
    save(fig, "t1_frame.png")


def progress(name, title, points, best_label, ylim, ylabel="public score", annotate=()):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = np.arange(len(points))
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=BLUE, lw=2, zorder=2)
    ax.scatter(xs, ys, s=46, color=BLUE, zorder=3, edgecolor=SURF, linewidth=2)
    ax.set_xticks(xs, [p[0] for p in points], fontsize=9, rotation=0)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    i = int(np.argmax(ys))
    ax.annotate(best_label, (xs[i], ys[i]), xytext=(-10, 14), textcoords="offset points", ha="right",
                fontsize=10.5, fontweight="semibold", color=INK)
    for item in annotate:
        k, lab, dy = item[0], item[1], item[-1]
        dx = item[2] if len(item) == 4 else 6
        ax.annotate(lab, (xs[k], ys[k]), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_title(title)
    save(fig, name)


def t1_progress():
    pts = [("Sat\n16:50", 0.453), ("18:10", 0.489), ("21:30", 0.594), ("21:54", 0.713), ("22:48", 0.738),
           ("23:10", 0.746), ("Sun\n00:47", 0.747), ("06:10", 0.747), ("07:36", 0.756), ("07:47", 0.771),
           ("07:57", 0.775), ("08:13", 0.787)]
    progress("t1_progress.png", "Task 1: public score over time (mean IoU)", pts,
             "0.787", (0.4, 0.82),
             annotate=[(0, "decoder heat-map + threshold", -26), (3, "first dense detector", -34),
                       (8, "larger frame", -38), (9, "+ second scale", -30)])


# ----------------------------------------------------------------------------------------------------------------
# Task 2
# ----------------------------------------------------------------------------------------------------------------
FAM_COL = {"QWEN_A": BLUE, "QWEN_B": ORANGE, "LLAMA": AQUA}
FAM_NAME = {"QWEN_A": "Qwen2.5-3B-Instruct", "QWEN_B": "Qwen2.5-1.5B-Instruct", "LLAMA": "Llama-3.2-1B-Instruct"}


def t2_partition():
    ev = {r["endpoint"]: r for r in csv.DictReader(open(ROOT / "task2/results/endpoint_evidence.csv"))}
    sub = {r["endpoint"]: r for r in csv.DictReader(open(ROOT / "task2/results/submission_final_0.85.csv"))}
    fam_of = {"A": "LLAMA", "B": "QWEN_A", "C": "QWEN_B"}

    def kind(e):
        v = ev[e]["verdict"]
        evd = ev[e]["evidence"].lower()
        if v.startswith("cosmetic:"):
            return {"system_prompt": "sys. prompt", "persona": "persona", "watermark": "watermark",
                    "temperature": "T = 1"}[v.split(":")[1]]
        if v == "variant:fp8":
            return "fp8"
        if v in ("variant:lora", "variant:strong"):
            return "LoRA"
        if v.startswith("judgement"):
            return "judgement"
        if v.startswith("score"):
            return "scores"
        # black-box split in QWEN_A: refine with the evidence text
        if "persona" in evd:
            return "persona"
        if "system prompt" in evd:
            return "sys. prompt"
        if "watermark" in evd:
            return "watermark"
        if "temperature" in evd:
            return "T = 1"
        if "lora" in evd:
            return "LoRA"
        return ""

    fig, ax = plt.subplots(figsize=(14, 5.8))
    bare(ax)
    order = ["QWEN_A", "QWEN_B", "LLAMA"]
    for fi, fam in enumerate(order):
        y0 = 2 - fi
        eps = sorted([e for e in sub if fam_of[sub[e]["family"]] == fam])
        main = sorted([e for e in eps if not sub[e]["model"][1:].startswith("s")])
        alone = sorted([e for e in eps if sub[e]["model"][1:].startswith("s")])
        col = FAM_COL[fam]
        ax.text(-0.3, y0 + 0.36, fam.replace("_", " "), fontsize=12, fontweight="semibold", va="bottom")
        ax.text(-0.3, y0 + 0.2, FAM_NAME[fam], fontsize=9.5, color=INK2, va="bottom")
        gx = 2.6
        w = 1.0
        ax.add_patch(FancyBboxPatch((gx - 0.08, y0 - 0.38), len(main) * w + 0.06, 0.76,
                                    boxstyle="round,pad=0.02,rounding_size=0.12", fc="none", ec=col, lw=2))
        ax.text(gx - 0.05, y0 + 0.46, "same model", fontsize=9, color=INK2)
        for k, e in enumerate(main):
            cx = gx + k * w + w / 2 - 0.05
            ax.add_patch(FancyBboxPatch((cx - 0.42, y0 - 0.3), 0.84, 0.6,
                                        boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="none"))
            ax.text(cx, y0 + 0.07, e, ha="center", va="center", color="white", fontsize=10, fontweight="semibold")
            ax.text(cx, y0 - 0.15, kind(e), ha="center", va="center", color="white", fontsize=7.5)
        ax2 = gx + len(main) * w + 0.6
        ax.text(ax2 - 0.05, y0 + 0.46, "separate models", fontsize=9, color=INK2)
        for k, e in enumerate(alone):
            cx = ax2 + k * w + w / 2 - 0.05
            ax.add_patch(FancyBboxPatch((cx - 0.42, y0 - 0.3), 0.84, 0.6,
                                        boxstyle="round,pad=0,rounding_size=0.08", fc=SURF, ec=col, lw=1.6))
            ax.text(cx, y0 + 0.07, e, ha="center", va="center", color=INK, fontsize=10, fontweight="semibold")
            ax.text(cx, y0 - 0.15, kind(e), ha="center", va="center", color=INK2, fontsize=7.5)
    ax.set_xlim(-0.4, 15.4)
    ax.set_ylim(-0.6, 3.1)
    ax.set_title("Task 2: final partition of the 33 endpoints", pad=14)
    fig.text(0.125, 0.04, "Under each endpoint, the evidence that placed it ('scores' and 'judgement' "
             "mark the 8 endpoints no test could settle; T = 1: temperature-1 decoding; blank: black-box split).",
             fontsize=9.5, color=INK2)
    save(fig, "t2_partition.png")


def t2_trap():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": 0.65, "width_ratios": [1, 1]})
    ax = axes[0]
    labels = ["system prompt\n(same model)", "fp8 quantisation\n(different model)"]
    vals = [0.730, 0.628]
    ax.barh([1, 0], vals, color=[ORANGE, BLUE], height=0.55)
    for yy, v in zip([1, 0], vals):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=10.5, color=INK2)
    ax.set_yticks([1, 0], labels)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("black-box distance from the plain model")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Black-box distance to the plain model (practice endpoints)")
    ax = axes[1]
    fams = ["QWEN_B", "QWEN_A", "ref (practice)"]
    right = [0.39, 0.35, 0.94]
    wrong = [0.82, 0.74, 1.41]
    names_r = ["Qwen2.5-1.5B", "Qwen2.5-3B", "SmolLM2-360M"]
    names_w = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "SmolLM2-1.7B"]
    y = np.arange(3)[::-1]
    ax.barh(y + 0.18, right, 0.34, color=AQUA, label="best candidate")
    ax.barh(y - 0.18, wrong, 0.34, color=GRAY, label="next candidate")
    for yy, v, n in zip(y, right, names_r):
        ax.text(v + 0.02, yy + 0.18, f"{v:.2f}  {n}", va="center", fontsize=9.5, color=INK2)
    for yy, v, n in zip(y, wrong, names_w):
        ax.text(v + 0.02, yy - 0.18, f"{v:.2f}  {n}", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y, fams)
    ax.set_xlim(0, 2.4)
    ax.set_xlabel("negative log-likelihood per token of the collected answers (lower = better fit)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Checkpoint identification by likelihood")
    save(fig, "t2_trap.png")


def t2_progress():
    pts = [("Sat\n21:45", 0.00), ("22:15", 0.05), ("22:45", 0.50), ("23:15", 0.10), ("Sun\n02:51", 0.60),
           ("03:22", 0.60), ("03:52", 0.60), ("07:54", 0.85)]
    progress("t2_progress.png", "Task 2: public score over time (diagnostic submissions excluded)", pts,
             "0.85", (-0.05, 1.0),
             annotate=[(2, "3 families found", 16), (3, "identity probes added", -30),
                       (4, "long collection", 16)])


# ----------------------------------------------------------------------------------------------------------------
# Task 3
# ----------------------------------------------------------------------------------------------------------------
def t3_idea():
    x = np.linspace(0, 14, 600)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    curves = [
        ("original model", 0.15, 0.12, INK2, "-"),
        ("gradient ascent", 9.5, 0.9, ORANGE, "--"),
        ("gold (retrained without the images)", 5.2, 4.5, BLUE, "-"),
    ]
    for lab, mu, sd, c, ls in curves:
        if sd > 3:
            y = norm.pdf(x, mu, sd) + 0.6 * norm.pdf(x, 0.4, 0.5) * 0.35
        else:
            y = norm.pdf(x, mu, sd)
        y = y / y.max()
        ax.plot(x, y, color=c, lw=2, ls=ls, label=lab)
        if c == BLUE:
            ax.fill_between(x, y, color=BLUE, alpha=0.12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("per-image loss on the 3,500 images to forget")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0), fontsize=10)
    ax.set_title("Per-image loss on the forget set")
    fig.text(0.125, -0.09, "Schematic shapes. The gold mean ≈ 5.2 and spread ≈ 4.5 are estimates from a model of the "
             "public scores, not measurements.", fontsize=9.5, color=INK2)
    save(fig, "t3_idea.png")


def t3_targets():
    u = np.linspace(0.001, 0.999, 400)
    mu, sigma = -2.5, 7.0
    m = mu - sigma * norm.ppf(u)
    m_ps = np.where(m > 0, 0.5 * m, m)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(u, m_ps, color=BLUE, lw=2)
    ax.fill_between(u, m_ps, 0, where=m_ps > 0, color=AQUA, alpha=0.18)
    ax.fill_between(u, m_ps, 0, where=m_ps < 0, color=ORANGE, alpha=0.15)
    ax.text(0.42, 3.5, "still correct,\nlower confidence", fontsize=9.5, color=INK2)
    ax.text(0.08, -19, "wrong class, margin\ngrows with difficulty", fontsize=9.5, color=INK2)
    for uu in (0.1, 0.5, 0.9):
        mm = mu - sigma * norm.ppf(uu)
        mm = 0.5 * mm if mm > 0 else mm
        ax.scatter([uu], [mm], color=BLUE, s=40, zorder=3, edgecolor=SURF, linewidth=2)
        ax.annotate(f"{mm:+.1f}", (uu, mm), xytext=(-34, -16), textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlabel("difficulty rank of the image  u  (0 = easiest, 1 = hardest)")
    ax.set_ylabel("target margin of the original class")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("m(u) = μ − σ·Φ⁻¹(u),  μ = −2.5,  σ = 7")
    ax = axes[1]
    classes = ["boat", "lantern", "figurine", "bird", "car"]
    orig = np.array([9.0, 2.0, 1.5, 0.5, 0.2])
    tgt = orig.copy()
    tgt[0] = orig[1:].max() - 2.5  # median image: margin −2.5
    xx = np.arange(len(classes))
    ax.bar(xx - 0.2, orig, 0.38, color=GRAY, label="original logits")
    ax.bar(xx + 0.2, tgt, 0.38, color=BLUE, label="target logits")
    ax.set_xticks(xx, classes)
    ax.set_ylabel("logit")
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Example target for one image")
    ax.text(0.98, 0.62, "example: a median image (u = 0.5)\nmargin set to −2.5 → now predicted 'lantern'",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=INK2)
    save(fig, "t3_targets.png")


def t3_progress():
    pts = [("Sat\n17:13", 0.024), ("22:58", 0.150), ("23:37", 0.259), ("Sun\n00:48", 0.352), ("03:02", 0.381),
           ("03:32", 0.448), ("06:06", 0.461), ("07:19", 0.465), ("07:59", 0.470)]
    progress("t3_progress.png", "Task 3: public score over time (forget × retain)", pts,
             "0.470", (0, 0.55),
             annotate=[(0, "output softening", 16, -8), (1, "SCRUB", 16, -8), (2, "rank-based targets", -30),
                       (4, "altered copies", -30)])


def t3_average():
    names = ["bt1", "y1", "bt2", "ps1", "st1"]
    vals = [0.459, 0.460, 0.460, 0.461, 0.465]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(names) + 1)[::-1]
    ax.scatter(vals, y[1:], s=90, color=GRAY, zorder=3, edgecolor=SURF, linewidth=2)
    ax.scatter([0.470], y[:1], s=120, color=BLUE, zorder=3, edgecolor=SURF, linewidth=2)
    for yy, v in zip(y[1:], vals):
        ax.text(v + 0.0006, yy, f"{v:.3f}", va="center", fontsize=9.5, color=INK2)
    ax.text(0.470 + 0.0006, y[0], "0.470", va="center", fontsize=10, fontweight="semibold")
    ax.set_yticks(y, ["average of the six\n(weights averaged)"] + [f"model {n}" for n in names])
    ax.set_xlim(0.455, 0.474)
    ax.set_xlabel("public score")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Weight average of six models and each model alone")
    fig.text(0.125, -0.1, "The sixth model (ps10, another seed of ps1) was not submitted alone.", fontsize=9.5,
             color=INK2)
    save(fig, "t3_average.png")


if __name__ == "__main__":
    t1_edits()
    t1_idea()
    t1_metric()
    t1_frame()
    t1_progress()
    t2_partition()
    t2_trap()
    t2_progress()
    t3_idea()
    t3_targets()
    t3_progress()
    t3_average()
