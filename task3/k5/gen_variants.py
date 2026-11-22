"""k5 — données tenues à l'écart SYNTHÉTIQUES (idée de sprintml/PostHocDatasetInference transposée aux images) :
variantes image-vers-image (diffusion, sd-turbo) des images à oublier : même contenu de classe, image que le modèle n'a jamais vue.
Usage (dossier unlearning/) : python k5/gen_variants.py --strength 0.5 --k 2 --shard 0/4 --out k5/var_s0.5_0.npz   [--n 16 --grid g.jpg pour un essai]"""
import argparse, os, sys, numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "k3", "pylib")); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--strength", type=float, default=0.5); ap.add_argument("--k", type=int, default=2); ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default="k5/var.npz"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--grid", default=""); ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--prompt", default=""); ap.add_argument("--bs", type=int, default=20)
a = ap.parse_args()
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda"); pipe.set_progress_bar_config(disable=True)
pipe.safety_checker = None
X = np.load(os.path.join(ROOT, "forget_images.npz"))["images"]; N = len(X)
i0, n0 = map(int, a.shard.split("/")); idx = np.arange(N)[i0::n0]
if a.n: idx = idx[:a.n]
out, src = [], []
g = torch.Generator(device="cuda").manual_seed(1234 + i0)
for v in range(a.k):
    for s in range(0, len(idx), a.bs):
        j = idx[s:s + a.bs]; x = torch.from_numpy(X[j]).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(512, 512), mode="bicubic", align_corners=False).clamp(0, 1)
        with torch.no_grad():
            y = pipe(prompt=[a.prompt] * len(j), image=x, num_inference_steps=a.steps, strength=a.strength, guidance_scale=0.0, output_type="pt", generator=g).images
        y = F.interpolate(y.float(), size=(224, 224), mode="area").clamp(0, 1)
        out.append((y * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()); src.append(j)
    print(f"variante {v + 1}/{a.k} faite", flush=True)
out = np.concatenate(out); src = np.concatenate(src)
np.savez(a.out, images=out, src=src); print("OK", a.out, out.shape)
if a.grid:
    from PIL import Image
    m = min(8, len(idx)); rows = [np.concatenate([X[idx[i]] for i in range(m)], 1)] + [np.concatenate([out[v * len(idx) + i] for i in range(m)], 1) for v in range(a.k)]
    Image.fromarray(np.concatenate(rows, 0)).resize((m * 160, (a.k + 1) * 160)).save(a.grid, quality=85)
