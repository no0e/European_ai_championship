# Rendu final — European AI Championship Paris (21-22/11/2026)

Cette branche ne contient que le code qui a produit le **meilleur score public** de chaque épreuve, avec toute la
chaîne depuis l'état initial (fichiers officiels, données brutes) jusqu'au fichier soumis. Chaque dossier a son
`README.md` : méthode, raisonnement, commandes dans l'ordre, ce qui a été écarté, limites.

| Épreuve | Meilleur score public | Soumission | Méthode en une ligne | Branche / commit d'origine |
|---|---|---|---|---|
| [`task1/`](task1/README.md) Localisation de watermark | **0,7872614154749301** | n° 3606, 22/11 08 h 13, `sub_m5.pt` | détecteur dense par méthode entraîné sur des composites exacts (tatoueurs officiels, **cadre tatoué plus grand que l'image**), deuxième échelle, affinage en CV, ensemble + filtre guidé + rectangles TrustMark | `task1-k5` @ `7ce9985` |
| [`task2/`](task2/README.md) Model Equality Testing | **0,85** | n° 3549, 22/11 07 h 54, `SOUMISSION_FINALE_0850.csv` | familles lues sur les habitudes du modèle de base, puis **boîte blanche** : checkpoints identifiés, watermark / température / fp8 / LoRA reconnus par teacher forcing des réponses déjà payées | `task2-kyllian` @ `57274af` (+ `9260647`) |
| [`task3/`](task3/README.md) Unlearning | **0,4700011591814447** | 22/11 07 h 59, `plat6.pt` | cibles « marge par rang » + distillation calée sur la perte par image, **moyenne de poids** de six modèles | `task3-k3` @ `1f4b369` |

## Organisation

```
src/        __init__.py (effets de bord Windows, `import src`), submit.py (journal des soumissions, utilisé par task1)
task1/      starter_kit officiel, signaux, rectangles, dense/ (v2 → v4, agent A), dense/k5/ (cadre, k6, assemblage m5)
task2/      sondes et collecte boîte noire, k/ (collecte longue, diagnostics, structures), wb2/ (boîte blanche), CSV final
task3/      fichiers officiels, ancres COCO, k3/ (cibles, distillation, moyenne), k5/ (variantes synthétiques)
```

## Ce qui n'est pas dans la branche

- **Données et poids** : images des épreuves, modèles officiels, checkpoints entraînés, réponses non versionnées,
  fichiers soumis `.pt`. Ils sont sur le scratch JURECA (`/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/`).
  Exception : pour la tâche 2, les réponses payées (`snapshot_*.jsonl`) et les CSV soumis sont inclus, car ils ne
  se rachètent pas.
- **Clés d'équipe** : jamais dans git (`WMLOC_API_KEY`, `MET_API_KEY` / `API_KEY`, `.submit_t3.py`).
- **Diagnostics et pistes abandonnées** : restés sur les branches d'origine ; les README les résument avec leurs mesures.
- Quelques commandes n'ont jamais été versionnées. Elles sont **reconstituées** et signalées comme telles dans les
  README (liste des runs assemblés pour m5, réglages de v7/v8 pour la tâche 1 ; fusion des variantes et cibles de `y1`
  pour la tâche 3).

## Ajouts propres à cette branche

- `task1/dense/k5/{common,assemble,compare_cv,submit_file}.py` : liens symboliques vers `task1/dense/`, parce que
  `asm.sh` et les scripts de `k5/` les appellent depuis `k5/`.
- `task2/` : la phase boîte noire est à la racine du dossier, comme dans `tasks/model_equality/` d'origine, pour que
  les chemins relatifs de `k/` et de `wb2/` restent valides.
- Aucun fichier de code n'a été modifié : tout est copié tel quel depuis les commits cités.
