"""Watermark signature: share of identical pairs among repeated samples of the same prompt, per endpoint.
On the reference endpoints the watermarked model (ref-06) repeats itself far more than the others (0.38 vs 0.04-0.13):
the watermark lowers the entropy of the sampled text without making it look like another model token by token.
Added for this repository (the value was computed inline during the hackathon; logged figures: ep-04 0.39,
ep-06 0.23, ep-10 0.17, ref-06 0.37 -- this script gives 0.36, 0.25, 0.20, 0.38 on all collected answers).
Usage: python src/whitebox/repetition.py [--out repetition.csv]   (reads $T2_WORK)"""
import argparse, collections, csv, itertools, json, os
from pathlib import Path

EXCL = {"sys_repeat", "sys_state"}          # identity / system-prompt probes: not samples of the model's text distribution
W = Path(os.environ.get("T2_WORK", "."))


def load(files):
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        if not (W / f).exists():
            continue
        for line in open(W / f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                D[r["endpoint"]][r.get("probe", r.get("id"))].append(r["response"])
    return D


def rate(P):
    pairs = same = 0
    for p, rs in P.items():
        if p in EXCL:
            continue
        for a, b in itertools.combinations(rs, 2):
            pairs += 1; same += a == b
    return same / pairs if pairs else float("nan"), pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="repetition.csv"); a = ap.parse_args()
    fam = json.load(open(W / "ep_families2.json"))
    rows = []
    for kind, files in (("ref", ["reference_answers.jsonl", "snapshot_answers_ref2.jsonl"]),
                        ("ep", ["answers_ep.jsonl", "answers_long_ep.jsonl"])):
        for e, P in sorted(load(files).items()):
            r, n = rate(P)
            rows.append(dict(endpoint=e, family=fam.get(e, "REF"), repeat_rate=round(r, 4), pairs=n))
    with open(W / a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for F in ("REF", "QWEN_A", "QWEN_B", "LLAMA"):
        sel = sorted((r for r in rows if r["family"] == F), key=lambda r: -r["repeat_rate"])
        print(f"{F:7s}", " ".join(f"{r['endpoint'][-2:]}:{r['repeat_rate']:.2f}" for r in sel))
    print("->", W / a.out)
