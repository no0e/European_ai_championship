"""Final partition of the 33 endpoints from the evidence table (results/endpoint_evidence.csv).

This script formalises, after the hackathon, how the 0.85 submission was built from the measured signals:
  - QWEN_A                        -> the black-box split (solve.py --taille 6), confirmed by the 01:19 diagnostic;
  - cosmetic:* (system prompt, persona, watermark, temperature 1)  -> main model of the family
    (the task says these do not change the model);
  - variant:* (fp8, LoRA, strong shift)                           -> alone (a different model);
  - score:in / score:out           -> structure inferred from the public diagnostic scores (fit.py, bayes.py);
  - judgement:in / judgement:out   -> no decisive evidence; the team's choice, kept explicit.
Each endpoint left alone gets its own model label. Usage (from $T2_WORK):
  python src/decide_partition.py --evidence results/endpoint_evidence.csv --blackbox submission_blackbox.csv \
         --out submission.csv [--check results/submission_final_0.85.csv]"""
import argparse, collections, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "structure"))
import mkcsv  # noqa: E402  (labels: family letter + group index, or <letter>s<nn> for an endpoint alone)

IN = ("cosmetic:", "score:in", "judgement:in")
OUT = ("variant:", "score:out", "judgement:out")


def blackbox_groups(path, family_letter):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["family"] == family_letter:
            by[r["model"]].append(r["endpoint"][-2:])
    return [sorted(g) for g in by.values() if len(g) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--blackbox", required=True, help="CSV of the black-box split (used for QWEN_A)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--check", default=None, help="reference CSV to compare with")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.evidence)))
    assert len(rows) == 33 and len({r["endpoint"] for r in rows}) == 33, "one row per endpoint"
    groups = {"QWEN_A": blackbox_groups(a.blackbox, mkcsv.L["QWEN_A"])}
    count = collections.Counter()
    for F in ("QWEN_B", "LLAMA"):
        main_group = []
        for r in rows:
            if r["family"] != F:
                continue
            v = r["verdict"]
            if v.startswith(IN):
                main_group.append(r["endpoint"][-2:])
            elif not v.startswith(OUT):
                raise SystemExit(f"unknown verdict {v!r} for {r['endpoint']}")
            count[v.split(":")[0]] += 1
        groups[F] = [sorted(main_group)]
    mkcsv.write(a.out, groups)
    print("main groups:", {F: g for F, g in groups.items()})
    print("verdicts (QWEN_B + LLAMA):", dict(count), "| judgement calls:",
          [r["endpoint"] for r in rows if r["verdict"].startswith("judgement")])
    print("->", a.out)
    if a.check:
        # Compared on content, not on bytes: csv.writer ends its lines with CRLF on Windows while the file
        # committed here keeps LF, so a byte comparison would report a difference that does not exist.
        def body(path):
            with open(path, 'rb') as f:
                return [ln.rstrip() for ln in f.read().splitlines() if ln.strip()]
        same = body(a.out) == body(a.check)
        print("identical to", a.check, ":", same)
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
