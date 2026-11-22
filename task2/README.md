# Tâche 2 : Model Equality Testing

**Meilleur score public : 0,85**, soumission n° 3549, 22/11 à 07 h 54.
Fichier soumis : `SOUMISSION_FINALE_0850.csv` (identique à `k/sub/wb_0755.csv` et à `best/20261122-075428_score0850.csv`).

Source : branche `task2-kyllian` (commit `57274af`). La partie boîte noire de la veille (familles, découpage de
QWEN_A) est reprise du commit `9260647`, celui qui a produit le CSV à 0,50. Journal complet : `HISTORY.md` de la
branche `task2-kyllian` (19 étapes, tous les scores).

---

## 1. Le problème

- **33 endpoints** (`ep-01` à `ep-33`), chacun sert un modèle de langage, interrogé en boîte noire (un prompt, un texte).
- Pour chacun : sa **famille** (modèle de base) et son **modèle** (mêmes poids). Ne changent **pas** le modèle :
  prompt système, persona, watermark, décodage. Le **changent** : quantification fp8, LoRA à force 0,3, 0,7 ou 1,5.
- **Score** : `0,1 × MCC_famille + 0,9 × MCC_modèle` sur les 528 paires, chaque MCC tronqué à 0.
- **Contraintes** : 100 réponses payées par endpoint, 256 tokens max, 4 requêtes simultanées pour l'équipe, une
  soumission toutes les 30 min. `temperature` et `seed` ignorés, messages système refusés.
- Ressource gratuite : 9 endpoints d'entraînement `ref-01` à `ref-09` à vérité connue (`practice_answers.csv`).

## 2. La partition finale et la preuve derrière chaque endpoint

Lettres de famille : A = LLAMA, B = QWEN_A, C = QWEN_B. Un groupe = même modèle ; tout endpoint hors groupe est seul.

| Famille (checkpoint identifié) | Groupe « modèle principal » | Endpoints seuls |
|---|---|---|
| **QWEN_A** (Qwen2.5-3B-Instruct) | 01, 02, 04, 12, 23, 33 | 14, 17, 18, 24, 25 |
| **QWEN_B** (Qwen2.5-1.5B-Instruct) | 03, 06, 07, 08, 20, 21, 30 | 22, 26, 28, 32 |
| **LLAMA** (Llama-3.2-1B-Instruct) | 10, 13, 16, 27, 29, 31 | 05, 09, 11, 15, 19 |

Ce qui place chaque endpoint (étapes 12 à 19 de `HISTORY.md`) :

- **Déguisements cosmétiques, donc dans le groupe** :
  - prompt système récité ou style « pas à pas » : `ep-02` (QWEN_A), `ep-08` (QWEN_B), `ep-29` (LLAMA) ;
  - persona tirée au hasard à chaque requête : `ep-23` (QWEN_A), `ep-20` (QWEN_B), `ep-16` (LLAMA) ;
  - **watermark** = taux de réponses répétées identiques élevé (`ref-06` : 37 % contre 5-15 %) : `ep-04` 39 %,
    `ep-06` 23 %, `ep-10` 17 %, un par famille. Le test boîte noire prenait `ep-06` et `ep-10` pour des modèles
    différents : le watermark ne déforme presque pas la distribution par token, mais il rend les répétitions identiques ;
  - **température 1** ajustée en boîte blanche (`wb2/prof2.py`, validé sur `ref-08` : T = 1,004) : `ep-33`, `ep-30`, `ep-31`.
- **fp8 isolés** : rapport de vraisemblance contre un fp8 simulé (e4m3, une échelle par tenseur ; `wb2/quant.py`,
  `wb2/llr.py`, validé sur `ref-02` : +19 nats contre −41 à −225 pour les 8 autres) : `ep-22` (+2 contre −39 à −79
  dans QWEN_B) et `ep-15` (+50 [20 ; 81] contre −23 à −91 dans LLAMA).
- **LoRA isolés** : corrélation des résidus de fréquence de tokens (`wb2/resid2.py`, validé : les trois LoRA des `ref`
  corrélées entre elles 0,31 à 0,75, le reste sous 0,2) : `ep-05`–`ep-11` (0,29) ; `ep-17`, `ep-18`, `ep-25`
  (0,24 à 0,37). Plus les variantes fortes vues par le test MMD par permutation (`k/fulltest.py`) : `ep-18`, `ep-25`,
  `ep-10` puis reclassé watermark, `ep-05`, `ep-28`, `ep-06` puis reclassé watermark, `ep-22`.
- **Membres restants du groupe, tranchés par les scores observés** : sous l'hypothèse « un groupe principal par
  famille », l'énumération des structures compatibles avec les sept scores de 1 h 19 à 3 h 52 (`k/fit.py`,
  `k/bayes.py`) donnait QWEN_B : 03, 07, 08, 20 sûrs, un seul de {21, 32} ; LLAMA : 16 et 29 sûrs, 31 à 0,79, 13 à
  0,69, 27 à 0,57. Le choix de `ep-21` plutôt que `ep-32` n'est pas justifié dans le journal.
- **QWEN_A** reprend tel quel le découpage du CSV à 0,50, confirmé par le diagnostic de 1 h 19 (0,55 avec QWEN_A seule
  découpée, ce qui correspond à un découpage juste).

Vérification faite pour ce rendu : `k/mkcsv.py` régénère `SOUMISSION_FINALE_0850.csv` **à l'octet près** à partir des
trois groupes ci-dessus, et les familles du CSV sont celles de `ep_families2.json` (MCC_famille = 1,00, mesuré à 23 h 45).

## 3. Le raisonnement, étape par étape

| Créneau | Public | Ce qui a changé, et pourquoi |
|---|---|---|
| 21/11 21 h 45 | 0,00 | Une seule famille : aucune paire « différente », MCC nul. |
| ≈ 22 h 15 | 0,05 | Familles d'après l'éditeur déclaré, modèles par distance. Une distance sur les sorties mesure d'abord le **cosmétique** : sur les `ref`, le prompt système (même modèle) est plus loin du modèle nu (0,730) que le fp8 (0,628). Aucun seuil ne corrige cet ordre inversé. |
| **22 h 45** | **0,50** | `families2.py` : **3 familles de 11** (LLAMA, QWEN_A, QWEN_B), lues sur des habitudes du modèle de base (réaction au préfixe d'injection, date de cutoff), pas sur l'éditeur cité ; « CLAUDE » était une persona. Hypothèse 6 cosmétiques + 5 variantes de poids par famille : `solve.py --taille 6` choisit le groupe de 6 le plus cohérent, endpoints à identité contradictoire forcés dedans. |
| 23 h 15 | 0,10 | Des sondes d'identité étaient entrées dans la distance : elles séparent le prompt système, pas les poids. Liste d'exclusion manuelle. |
| 23 h 45 | 0,10 | Essai volontaire, chaque famille sous un seul label : score = 0,1 × MCC_famille ⇒ **MCC_famille = 1,00**. Tout le reste est dans la partie modèle. |
| 22/11 1 h 19 → 2 h 21 | 0,55 ; 0,35 ; 0,10 | **Diagnostics** : une famille découpée comme à 0,50, les deux autres en « tout différent ». QWEN_A juste, QWEN_B faux d'un échange, LLAMA sans valeur. |
| 2 h 51 → 3 h 52 | 0,60 (×3) | Collecte longue (`k/plan2.py`, 44 requêtes à 256 tokens par endpoint : questions fragiles, choix d'outils équivalents, choix arbitraires, suites aléatoires) ; partitions de QWEN_B et LLAMA tirées des réponses. Erreur de lecture corrigée (`ep-07` répondait dans une autre forme). « Indiscernable au test » ne veut pas dire « même modèle ». |
| 6 h 20 | 0,50 | Essai d'une autre branche (`ep-22` dans le groupe de QWEN_B) : neutre. |
| **7 h 54** | **0,85** | **Boîte blanche sur GPU** (`wb2/`) : checkpoints identifiés, puis watermark, température 1, fp8 et LoRA reconnus endpoint par endpoint (section 2). |

Principe tenu tout du long : le leaderboard sert de **diagnostic**, pas de sélecteur (écart type ≈ 0,04 à 0,07 ;
`PROTOCOLE.md` sur la branche source). À partir de 3 h 52, les scores ont servi à trancher entre des structures que les
données ne départageaient pas : c'est dans les règles, mais ce n'est pas une méthode autonome.

### Pourquoi la boîte blanche a tout débloqué

- En boîte noire, fp8 et LoRA 0,3 sont indétectables (mesures sur les `ref`, et arXiv:2506.06975), et le watermark
  ressemble à un autre modèle. En **teacher forcing** sous le bon checkpoint local, chaque réponse déjà payée donne une
  log-vraisemblance par token : on peut alors tester des hypothèses précises (température, fp8 simulé, résidus).
- Identification des checkpoints (`wb2/score.py` + `wb2/ident.py`, NLL par token des réponses, gabarit de chat) :
  QWEN_B = Qwen2.5-1.5B-Instruct (0,39 ; 0.5B : 0,82), QWEN_A = Qwen2.5-3B-Instruct (0,35 ; 1.5B : 0,74),
  LLAMA = Llama-3.2-1B-Instruct avec l'en-tête système daté (0,63), `ref` = SmolLM2-360M-Instruct (0,94).
- Le serveur échantillonne en température pure : aucune hypothèse top-p, top-k ou pénalité de répétition ne tient
  (`wb2/dec.py`).
- Chaque test a d'abord été **validé sur les `ref`** (vérité connue), puis appliqué aux `ep`.
- Aucune requête payante n'a été dépensée pour la boîte blanche : elle relit les réponses déjà collectées.

## 4. Pipeline complète, de l'état initial au CSV à 0,85

### Étape 0 — environnement

```bash
cd task2
pip install -r requirements.txt                 # + transformers, torch pour wb2/
export PYTHONPATH=$(cd .. && pwd)               # les scripts font `import src` (dossier src/ à la racine de la branche)
export MET_API_KEY=...                          # collect.py ; k/api.py lit API_KEY (jamais dans git)
```

### Étape 1 — sondes, collecte boîte noire, familles, découpage de QWEN_A (21/11)

```bash
python query_reference.py                       # réponses gratuites des ref-* (reference_answers.jsonl)
python collect.py --endpoints ep --only-file ep_pass1.txt --repeats 4    # passes ep_ident.txt, ep_pass1..5c.txt, sondes de probes.py
cp snapshot_answers_ep.jsonl answers_ep.jsonl   # ou repartir des 1 672 réponses payées versionnées
python families2.py                             # -> ep_families2.json (3 familles de 11)
python solve.py --taille 6 --out submission_solve.csv
```
Le CSV à 0,50 (`best/20261121-233444_score0500.csv`) a été produit par `solve.py --taille 6` sur les **966 premières**
réponses (le snapshot en contient 1 672) : le rejouer sur tout le snapshot peut donner un autre découpage. Seul le
découpage de QWEN_A en est repris.

### Étape 2 — collecte longue, diagnostics, structures compatibles (22/11, `k/`)

```bash
cd k
python plan2.py                                 # simulation du plan (rien n'est envoyé)
python plan2.py pilot --go && python plan2.py full --go     # -> ../answers_long_ep.jsonl (1 453 réponses)
cp ../snapshot_answers_long_ep.jsonl ../answers_long_ep.jsonl   # pour repartir des réponses versionnées
python analyze.py QWEN_B LLAMA ; python fulltest.py QWEN_A QWEN_B LLAMA ; python dose.py ; python agree.py
python submit.py sub/diag_QWEN_A.csv "diagnostic"            # idem diag_QWEN_B, diag_LLAMA, cand_0251, cand_0322, cand_0352
python fit.py ; python bayes.py                 # structures compatibles avec obs.txt, partition de score attendu maximal
```
`obs.txt` liste les CSV soumis (`../best/`) et leurs scores ; `fit.py` et `bayes.py` tournent hors ligne (vérifié pour
ce rendu, en quelques secondes sur CPU).

### Étape 3 — boîte blanche sur GPU (`wb2/`)

Disposition attendue par `wb2/score.py` (variable `T2W`, par défaut `~/t2wb`) :
```
~/t2wb/code/   <- wb2/*.py, wb2/*.sh
~/t2wb/data/   <- ep_families2.json, snapshot_answers_ep.jsonl, answers_long_ep.jsonl,
                  reference_answers.jsonl, snapshot_answers_ref2.jsonl
~/t2wb/models/ <- Qwen--Qwen2.5-1.5B-Instruct, Qwen--Qwen2.5-3B-Instruct, unsloth--Llama-3.2-1B-Instruct,
                  HuggingFaceTB--SmolLM2-360M-Instruct (téléchargés depuis Hugging Face, lus hors ligne)
~/t2wb/out/    <- pickles de score.py
```
```bash
cd ~/t2wb/code
bash chain_a.sh; bash chain_b.sh; bash chain_c.sh; bash chain_d.sh; bash chain_e.sh
#   score.py : NLL des réponses sous la base (--dec), sous des fp8 simulés (--quant fp8t…), masse des tokens fréquents (--watch)
python ident.py ../out/*__dec.pkl ../out/*__tmpl.pkl                     # quel checkpoint (NLL, rang 1)
python dec.py  ../out/Qwen2.5-1.5B-Instruct__ep__dec.pkl QWEN_B          # hypothèses de décodage
python prof2.py ../out/Qwen2.5-1.5B-Instruct__ep__dec.pkl QWEN_B 0.7     # température ajustée par endpoint
python llr.py  ../out/Qwen2.5-1.5B-Instruct__ep__dec.pkl ../out/Qwen2.5-1.5B-Instruct__ep__fp8t.pkl 0.7 QWEN_B   # fp8
python resid2.py ../out/Qwen2.5-1.5B-Instruct__ep__watch.pkl QWEN_B     # résidus de fréquence de tokens (LoRA)
# idem pour LLAMA (Llama-3.2-1B, --sys date), QWEN_A (Qwen2.5-3B) et les ref (SmolLM2-360M, validation)
```
`run_gpu.sh` lance chaque calcul comme étape `srun --overlap` d'un job déjà alloué de l'équipe (aucun nouveau nœud) ;
remplacer `J=` par un job en cours. Le taux de répétitions identiques (signature du watermark) a été calculé
directement sur les réponses, sans script versionné : part des tirages d'une même sonde dont le texte est identique à
un autre tirage, par endpoint.

### Étape 4 — CSV final et soumission

```bash
cd k
python -c "import mkcsv; mkcsv.write('sub/wb_0755.csv', {
  'QWEN_A': [['01','02','04','12','23','33']],
  'QWEN_B': [['03','06','07','08','20','21','30']],
  'LLAMA':  [['10','13','16','27','29','31']]})"
python submit.py sub/wb_0755.csv "boîte blanche v1"      # archive aussi dans ../best/
```

## 5. Ce qui n'a pas marché (mesuré)

- Moyenne des 49 sondes : MCC 0,000 (3 bonnes sondes : 0,446). Grouper plusieurs questions par requête : −44 % de
  pouvoir discriminant. Normaliser fort les réponses : moins de signal.
- Seuil au plus grand saut entre distances : MCC 0. Sondes de choix arbitraire généralisées : 4 marges sur 12 négatives.
- « Gros écart de style = cosmétique » : faux (les plus gros écarts sont les LoRA 1,5 et 0,7).
- Test « base à une température près » en boîte noire : sans puissance. Latence comme canal auxiliaire : sans lien.
- z-score KGW avec la clé par défaut de `transformers` : −0,4 sur `ref-06`, abandonné.
- Regrouper les 8 endpoints LLAMA indiscernables au test (3 h 52) : 0,60, des faux « même » compensent les vrais.
- Isoler les endpoints « Claude » comme LoRA (branche `task2-whitebox`, 7 h 22) : 0,15, ce sont des personas.

## 6. Fichiers

| Fichier | Rôle |
|---|---|
| `task_template.py`, `requirements.txt`, `practice_answers.csv`, `score_practice.py` | fichiers officiels de l'épreuve (vérité et score des `ref`) |
| `query_reference.py`, `reference_prompts.txt`, `reference_answers.jsonl`, `snapshot_answers_ref2.jsonl` | réponses des `ref-*` (gratuites) |
| `probes.py`, `collect.py`, `ep_ident.txt`, `ep_pass*.txt` | sondes et collecte boîte noire (cache, coût chiffré avant envoi) |
| `snapshot_answers_ep.jsonl` | les 1 672 réponses payées de la collecte boîte noire |
| `analyze.py`, `build_submission.py`, `families2.py`, `solve.py`, `ep_features.json` | lectures des réponses, familles, découpage par famille (version du commit `9260647`) |
| `ep_families2.json` | familles des 33 endpoints (MCC_famille = 1,00) |
| `archive.py`, `best/` | archivage de chaque soumission ; les 9 CSV soumis relus par `k/fit.py` |
| `k/api.py`, `k/plan.py`, `k/plan2.py`, `k/collect_full.log` | client sans réessai sur 502, plan et journal de la collecte longue |
| `snapshot_answers_long_ep.jsonl` | les 1 453 réponses longues payées |
| `k/parse.py`, `k/data.py`, `k/analyze.py`, `k/agree.py`, `k/sims.py`, `k/seqstats.py`, `k/fulltest.py`, `k/dose.py` | lecture des batteries, tests par permutation, axe dose-effet |
| `k/fit.py`, `k/bayes.py`, `k/obs.txt` | structures compatibles avec les scores observés, a posteriori |
| `k/mkcsv.py`, `k/submit.py`, `k/sub/` | écriture et soumission des CSV (diagnostics, candidats, final) |
| `wb2/score.py`, `wb2/quant.py` | teacher forcing sous un checkpoint local ; fp8 / int8 simulés |
| `wb2/ident.py`, `wb2/dec.py`, `wb2/prof2.py`, `wb2/llr.py`, `wb2/resid2.py` | checkpoint, décodage, température, fp8, LoRA |
| `wb2/run_gpu.sh`, `wb2/chain_[a-e].sh` | lanceurs GPU |
| `SOUMISSION_FINALE_0850.csv` | le CSV à 0,85 |

## 7. Limites

- Le choix `ep-21` contre `ep-32` dans QWEN_B n'est pas documenté ; `ep-09`, `ep-19` (LLAMA), `ep-24` et `ep-26` sont
  seuls faute de preuve d'appartenance, pas sur une preuve de différence.
- Le score public couvre une partie des paires : 0,85 n'est pas garanti sur les paires cachées.
- Hors de ce rendu, car postérieurs au 0,85 ou sans effet sur lui : `wb2/xfit.py` (test croisé appris),
  `wb2/design.py` et la dernière collecte à haute entropie, le travail parallèle de la branche `task2-whitebox`
  (confronté après coup à cette partition, il propose la même dans `whitebox/partition_proposee.csv`).
