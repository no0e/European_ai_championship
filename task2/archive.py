#!/usr/bin/env python3
"""Archive chaque soumission notée et garde la meilleure à l'abri d'une réécriture.

    python archive.py --csv submission_auto.csv --score 0.5 --note "isoler 3, 1482 reponses"
    python archive.py --liste

Pourquoi ce fichier existe. Le serveur retient la MEILLEURE soumission de l'équipe (« the lowest
and highest best scores of all teams on that task »), donc une soumission exploratoire ne peut
pas faire reculer le classement — mais le CSV lui-même est écrasé à chaque créneau par le
générateur. La soumission à 0,50 de 22 h 45 a ainsi été perdue et n'a pu être récupérée que dans
l'historique git, par chance. Les suivantes sont copiées ici avant tout écrasement.

`best/BEST.csv` est toujours la meilleure connue : c'est elle qu'on resoumet si une piste tourne
mal et qu'il faut revenir au point haut, et c'est elle qu'on montre au jury.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOSSIER = HERE / "best"
INDEX = DOSSIER / "index.jsonl"
MEILLEURE = DOSSIER / "BEST.csv"
META = DOSSIER / "BEST.json"


def entrees():
    if not INDEX.exists():
        return []
    return [json.loads(l) for l in INDEX.open(encoding="utf-8") if l.strip()]


def archiver(csv: Path, score, note: str = "") -> dict:
    """Copie la soumission, met à jour BEST.csv si elle est la meilleure. -> l'entrée écrite."""
    DOSSIER.mkdir(exist_ok=True)
    horo = datetime.now().strftime("%Y%m%d-%H%M%S")
    etiquette = "x" if score is None else f"{float(score):.3f}".replace(".", "")
    copie = DOSSIER / f"{horo}_score{etiquette}.csv"
    shutil.copyfile(csv, copie)
    entree = {"t": datetime.now().isoformat(timespec="seconds"), "fichier": copie.name,
              "score": score, "note": note}
    with INDEX.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entree, ensure_ascii=False) + "\n")

    anciens = [e["score"] for e in entrees()
               if isinstance(e.get("score"), (int, float)) and e["fichier"] != copie.name]
    record = max(anciens) if anciens else None
    if isinstance(score, (int, float)) and (record is None or score > record):
        shutil.copyfile(csv, MEILLEURE)
        META.write_text(json.dumps({**entree, "precedent_record": record},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        entree["nouveau_record"] = True
    return entree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--score", type=float, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--liste", action="store_true")
    a = ap.parse_args()
    if a.liste:
        es = entrees()
        if not es:
            print("aucune soumission archivée")
            return 0
        for e in es:
            s = e.get("score")
            print(f"  {e['t']}  score {('%.3f' % s) if isinstance(s, (int, float)) else '  ?  '}"
                  f"  {e['fichier']}  {e.get('note', '')}")
        best = max((e for e in es if isinstance(e.get("score"), (int, float))),
                   key=lambda e: e["score"], default=None)
        if best:
            print(f"\n  meilleure : {best['score']:.3f}  ({best['fichier']}) -> best/BEST.csv")
        return 0
    if not a.csv:
        print("il faut --csv ou --liste", file=sys.stderr)
        return 2
    e = archiver(Path(a.csv), a.score, a.note)
    print(f"archivé : {e['fichier']}" + ("  (NOUVEAU RECORD -> BEST.csv)"
                                         if e.get("nouveau_record") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
