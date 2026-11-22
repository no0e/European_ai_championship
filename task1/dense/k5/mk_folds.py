"""Vue « par plis » d un run sans plis (base synthétique) : mix/<nom>/f<k>/ = prédictions train des images du pli k
+ prédictions test complètes. Sans cela, l assemblage pèse ce run différemment sur le train et sur le test
(constat de l agent A, 22/11 06 h 10)."""
import sys, os, numpy as np
from pathlib import Path
from common import METHODS, load_meta
from t1_metric import stratified_folds
run, name = Path(sys.argv[1]), sys.argv[2]
allm = load_meta("train"); folds = stratified_folds([m["method"] for m in allm], 5, 0)
fold = {m["image_name"]: f for m, f in zip(allm, folds)}
for meth in METHODS:
    f = run / f"{meth}_train.npz"
    if not f.exists(): continue
    z = np.load(f); names = z["name"].tolist(); probs = z["probs"]
    for k in range(5):
        d = Path("mix") / name / f"f{k}"; d.mkdir(parents=True, exist_ok=True)
        sel = [i for i, n in enumerate(names) if fold[n] == k]
        np.savez_compressed(d / f"{meth}_train.npz", probs=probs[sel], name=np.array([names[i] for i in sel]))
        t = d / f"{meth}_test.npz"
        if t.is_symlink() or t.exists(): t.unlink()
        os.symlink((run / f"{meth}_test.npz").resolve(), t)
    print(meth, "->", f"mix/{name}/f0..f4")
