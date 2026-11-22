"""Écrit un CSV de soumission à partir de groupes par famille (le reste des endpoints est laissé seul)."""
import csv, json, os, sys
from pathlib import Path
T = Path(os.environ.get('T2_WORK', Path(__file__).resolve().parent.parent))   # working dir (config.env)
FAM = json.load(open(T / 'ep_families2.json')); L = {'LLAMA': 'A', 'QWEN_A': 'B', 'QWEN_B': 'C'}
QA = [['ep-01', 'ep-02', 'ep-04', 'ep-12', 'ep-23', 'ep-33']]
def write(path, groups):
    rows = []
    for i in range(1, 34):
        e = f'ep-{i:02d}'; f = FAM[e]; lab = None
        for gi, g in enumerate(groups.get(f, [])):
            g = [x if x.startswith('ep-') else f'ep-{x}' for x in g]
            if e in g: lab = f'{L[f]}{gi + 1}'
        rows.append((e, L[f], lab or f'{L[f]}s{i:02d}'))
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['endpoint', 'family', 'model']); w.writerows(rows)
    return rows
