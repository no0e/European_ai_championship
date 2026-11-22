# Tâche 1 : localisation de watermark dans des images

**Meilleur score public : 0,7872614154749301** (affiché 0,7873), soumission n° 3606, 22/11 à 08 h 13.
Fichier soumis : `dense/k5/out/sub_m5.pt` (sha1 `317a0ead1e`), assemblage « m5 ». CV 5 plis : 0,7723.

Source : branche `task1-k5` (commit `7ce9985`), qui ajoute `dense/k5/` à la branche `task1-final` / `main`.
Les journaux complets de la tâche restent sur ces branches : `HISTORY.md`, `JOURNAL_TACHE1.md` (une ligne par
mesure), `dense/DECISIONS_ET_RESULTATS.md`, `history/HISTORY.md` (journée du 22/11).

---

## 1. Le problème

- **1 020 images de test** 512×512, un tiers par méthode de watermark : **PixelSeal** (256 bits), **TrustMark**
  (100 bits), **StegaStamp** (100 bits). La méthode **et le message** sont fournis pour chaque image.
- Il faut prédire, sur une grille **128×128** (cellules de 4×4 px), les cellules tatouées (1 si au moins 50 % de la
  cellule l'est).
- **Score** : mIoU par image, (IoU₀ + IoU₁) / 2, puis moyenne sur les images. Une prédiction constante plafonne à 0,5.
- Fabrication des images (énoncé) : le watermark est posé sur une zone, puis une retouche parmi quatre :
  outpainting (zone = rectangle), suppression ou ajout d'objet (zone = toute l'image moins un trou),
  transplantation d'un objet tatoué (zone = forme de l'objet).
- **Train** : 510 images (170 par méthode) avec masque, méthode, message et `full_watermark`. Le kit officiel fournit
  les tatoueurs **exacts** (`embed`) et les décodeurs.
- Deux régimes, exactement 85/85 par méthode (`figures/resume_train.txt` sur `main`) : `full` (image entière tatouée
  puis des trous, fraction tatouée médiane 0,94) et non-full (petite zone, médiane 0,04 ; un rectangle une fois sur
  deux, un objet sinon).
- Soumission : `.pt` `{"image_names", "masks"}`, masks `[1020, 128, 128]` booléens. Public sur 30 % du test.

## 2. L'idée de la méthode finale

**Constat fondateur (mesuré sur le train)** : dans les images *full* et dans les objets transplantés, le tatouage est
à l'échelle de l'image entière, puis l'image est masquée :

    y = M · embed(x, m) + (1 − M) · x

On peut donc **fabriquer autant d'exemples exacts que l'on veut** avec les tatoueurs officiels du kit, et entraîner,
pour chaque méthode, un **détecteur dense** « tatoué / pas tatoué » qui n'a pas besoin du message.

Les briques, dans l'ordre où elles ont été ajoutées :

1. **Détecteur dense par méthode** (`dense/train_dense.py`, `dense/common.py`) : le squelette est le **décodeur du
   watermark lui-même** (PixelSeal : ConvNeXtV2 ; TrustMark : ResNet-50) ou un ConvNeXt-tiny ImageNet (StegaStamp),
   suivi d'un FPN et d'une branche pleine résolution ; sortie 128×128. Partir du décodeur est indispensable : un
   squelette générique n'apprend rien sur PixelSeal (AUC 0,5).
2. **Composites synthétiques** (`dense/gen_pairs.py`) : paires (x, embed(x, m)) sur des images du même domaine
   (images non-full **des deux autres méthodes**, recadrages DIV2K), puis masques à la volée : vrais masques du train
   transformés, segments SAM, formes aléatoires, distribution réelle du nombre et de l'aire des trous (A1a).
3. **Affinage en validation croisée sur les vraies images du train** (v3), puis un tour de **pseudo-étiquettes** des
   images de test (v4) : le modèle du pli k pseudo-étiquette le test, et seul un modèle qui n'a pas vu le pli k s'en sert.
4. **Assemblage** (`dense/assemble.py`) : moyenne des probabilités des modèles ; **filtre guidé** par l'image couleur
   (8 réglages) et règle de décision (seuil, top-k, rectangle englobant, nettoyage des petites composantes) choisis
   **hors pli** par (méthode, régime prédit) ; le régime vient de la log-vraisemblance globale du message
   (`compute_signals.py`, juste à 97 %).
5. **Rectangles TrustMark** (`rect_search.py`) : sur les images test TrustMark prédites non-full, recherche du recadrage
   qui relit le message ; 75 rectangles acceptés (précision par bit ≥ 0,90) remplacent la carte.
6. **Deuxième échelle (A3)** de l'agent A : carte fine 128 au lieu de 64 dans le détecteur (`--scale2`), entraînement
   long sur un réservoir doublé (`--with_full`), amplitude du tatouage variable (`--alpha 0.8 1.2`).
7. **Le cadre tatoué est plus grand que l'image (k5, 22/11 matin)** : c'était l'écart synthétique / réel. Les paires
   sont désormais tatouées sur un cadre de côté s × 512 (s entre 1,02 et 1,18), complété par réflexion, puis recadré
   (`dense/k5/gen_pairs.py --frame 0.84 --frame_max 1.18`).
8. **Assemblage m5** : modèles « deuxième échelle + cadre » (k6), leurs bases (k6b), base « cadre » k5b, modèles
   deuxième échelle de l'agent A (v6, v6p, v7, v7p, v8) ; StegaStamp : v3, v4, base cadre k5b et k5.

## 3. Le raisonnement, étape par étape

| Soumission | Public | CV 5 plis | Ce qui a changé, et pourquoi |
|---|---|---|---|
| R1, n° 1081 (21/11 16 h 50) | 0,4531 | 0,4621 | Carte 8×8 du décodeur (logits par position × signe du message) + seuil par méthode. Diagnostic : deviner le régime vaut +0,17 à lui seul (oracle régime 0,4547) ; le maximum de la carte 8×8 tombe dans le rectangle 4 % du temps : **la carte 8×8 ne localise pas**. |
| R3, n° 1155 (18 h 10) | 0,4890 | — | Décoder le **recadrage exact** d'un rectangle relit le message (TrustMark 1,00, StegaStamp 0,97, PixelSeal 0,66) ⇒ `rect_search.py` sur les 172 images test TrustMark prédites non-full : 75 rectangles acceptés (précision ≥ 0,90). |
| (non soumis) | — | 0,4654 | Segmenteur U-Net sur RGB (`seg/`, branche `main`) : il n'apprend que le régime. Le watermark n'est pas visible dans les pixels. |
| v1, n° 1660 | 0,594 | — | Premier détecteur dense. Bug : les images de base incluaient la méthode entraînée, donc de vraies zones tatouées étiquetées « propres ». |
| **v2**, n° 1694 (21 h 54) | **0,713** | 0,6765 | Images de base = non-full des **deux autres** méthodes. PS 0,611, TM 0,592, SS 0,826 en CV. |
| v3, n° 1820 | 0,7381 | 0,7085 | Affinage en CV avec les vraies images du train (+0,032 apparié). |
| v3 + filtre guidé, n° 1883 | 0,7463 | 0,7162 | Filtre guidé couleur de la carte, réglage choisi hors pli (+0,0077 ; objets PS +0,029). |
| v3 + v4, n° 2170 (00 h 47) | 0,7469 | 0,7213 | v4 = pseudo-étiquettes du test + zone non tatouée prise dans une autre image (`--p_fill`) ou lissée (`--p_blur`), 3 000 pas ; moyenne v3 + v4 (+0,0051). |
| v3 + v4 + A3, n° 3173 (06 h 10) | 0,7467 | 0,7431 (fausse) | Les cartes A3 ajoutées **sans plis** pesaient 43 % sur le train et 13 % sur le test : la CV évaluait un autre ensemble que celui soumis. **Correctif : tout modèle sans plis est découpé en vue par plis** (`dense/k5/mk_folds.py`) avant l'assemblage. |
| m1_sel, n° 3491 (07 h 36) | 0,7555 | 0,7298 | Premiers modèles entraînés sur le synthétique **avec cadre**. |
| m2b, n° 3530 (07 h 47) | 0,7714 | 0,7599 | + modèles deuxième échelle de l'agent A (v6, v6p, v7, v7p, v8). |
| m3, n° 3562 (07 h 57) | 0,7754 | 0,7632 | + bases deuxième échelle avec cadre (k6b), StegaStamp avec cadre. |
| **m5, n° 3606 (08 h 13)** | **0,7873** | **0,7723** | Modèles deuxième échelle + cadre **affinés en CV** (k6), bases et affinage StegaStamp avec cadre. |

Toutes les décisions ont été prises sur la **CV 5 plis appariée** (différence par image + bootstrap 95 %,
`dense/compare_cv.py`), jamais sur le public (30 % du test, ±0,005 de bruit). Le public a suivi la CV à environ
0,7 fois le gain.

### Comment le cadre a été trouvé (k5)

- Le motif de message recalculé `embed(y, m) − embed(y, m̄)` ne corrèle avec l'image réelle que si on complète
  l'image par réflexion jusqu'à un cadre de côté s × 512, s entre 1,02 et 1,17, avec un décalage quelconque ; environ
  16 % des images sont à s = 1. Sur TrustMark, la statistique Z passe de ≈ 0 à +70…+120 au bon cadre ; le cadre est
  retrouvé sur 82 % des images full du train (script de diagnostic `dense/k5/frames.py`, branche `task1-k5`).
- Nos composites tatouaient le cadre 512 entier : c'était l'écart synthétique / réel mesuré depuis la veille (A1 :
  0,711 en synthétique contre 0,561 en réel sur les trous PixelSeal).
- Avec `gen_pairs.py --frame`, une base entraînée **sans aucune vraie image** fait mieux que l'ancien ensemble affiné
  sur le réel (PixelSeal full 0,667 contre 0,647, TrustMark full 0,657 contre 0,634).
- Deuxième échelle + cadre s'additionnent : PixelSeal full ≈ 0,71, TrustMark full ≈ 0,68, objets ≈ 0,74,
  StegaStamp full ≈ 0,94 en CV.

## 4. Pipeline complète, de l'état initial à `sub_m5.pt`

Les scripts ont tourné sur JURECA (4 × A100 par nœud) avec `T=/p/scratch/training2666/diette1/Sevel_Labs_ENSAI/Hackathon_Watermark_Localisation_Task`.
Correspondance : `$T` = ce dossier `task1/` ; `$T/dense` = `task1/dense/` ; `$T/dense/k5` = `task1/dense/k5/` ;
`$T/agentA` = un second dossier de travail contenant une copie du code de `dense/` (l'agent A y a lancé `job_long.sh`
et `job_cv_long.sh`, avec des liens `runs/v2`, `runs/v3`, `runs/v4` vers les runs de `dense/`) ;
`$T/paulin` = dossier où `rect_search.py` a écrit ses sorties. Les chemins sont écrits en dur dans les `.sh` : les
adapter avant de relancer ailleurs.

### Étape 0 — environnement et données (hors git)

```bash
cd task1
pip install -r starter_kit/requirements.txt          # + timm, scipy, torchvision (utilisés par dense/)
python starter_kit/download_models.py               # tatoueurs/décodeurs officiels -> starter_kit/models/
python -c "from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"   # depuis starter_kit/
```
- `release/{train,test}/` : images, `metadata.json`, `train/masks/*.png`, `train/masks_128.npz`.
- `clean/DIV2K_valid_HR/` : images DIV2K (validation HR), utilisées par `gen_pairs.py` pour quelques recadrages.
- `dense/weights/convnext_tiny.safetensors` : poids ImageNet timm de ConvNeXt-tiny (squelette StegaStamp). La
  provenance exacte du fichier n'est pas consignée.
- `sam/pylib` : bibliothèques ajoutées au `PYTHONPATH` sur le cluster (dont `segment_anything`).

### Étape 1 — signaux globaux des décodeurs (régime full / non-full)

```bash
python compute_signals.py --split train
python compute_signals.py --split test          # -> release/signals_{train,test}.npz (global_ll, map8)
```
`dense/common.py::regime_thresholds()` apprend sur le train un seuil par méthode sur `global_ll` (> seuil = full).

### Étape 2 — rectangles TrustMark du test (job 15901771, 4 GPU, ≈ 18 min)

```bash
# images test TrustMark prédites non-full (172), une par ligne, dans names_tm_test.txt
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python rect_search.py --split test --method trustmark \
   --names names_tm_test.txt --amp --shard $i/4 --out paulin/out/rect_tm_test_s$i.json & done; wait
```
Commande **reconstituée** à partir des options du script et du journal (le lanceur n'est pas versionné). `assemble.py`
remplace la carte par le rectangle si la précision par bit est ≥ 0,90 et l'image prédite non-full (75 images).

### Étape 3 — segments SAM (pour les composites)

```bash
cd sam && sbatch job_sam.sh                       # train ; puis SPLIT=test : sbatch --export=ALL,SPLIT=test job_sam.sh
```
`MaskSampler` (`dense/common.py`) lit `sam/out/{train,test}/<image>.npz` s'ils existent, pour fabriquer des masques
en forme d'objet. Sans ces fichiers, il se rabat sur les autres formes.

### Étape 4 — v2 : paires synthétiques et détecteur de base (`dense/job_dense.sh`)

```bash
cd dense && mkdir -p out && sbatch --export=ALL,RUN=v2 job_dense.sh
#  -> pairs/<méthode>/ (gen_pairs.py --reps 2 --rects 2), runs/v2/<méthode>.pt (6 000 pas, --p_rect 0.12),
#     runs/v2_b/trustmark.pt (2e graine TrustMark)
```

### Étape 5 — v3 : affinage en CV 5 plis sur les vraies images (`dense/job_cv.sh`)

```bash
sbatch --export=ALL,RUN=v3,INIT=v2 job_cv.sh       # -> runs/v3/f{0..4}/, runs/v3_b/f{0..4}/ (1 500 pas par pli)
```
Chaque pli k : départ des poids v2, lots moitié synthétiques, moitié vraies images des 4 autres plis ; sorties
`<méthode>_train.npz` (probabilités hors pli) et `<méthode>_test.npz`.

### Étape 6 — v4 : pseudo-étiquettes du test (`dense/job_v4.sh`)

```bash
sbatch job_v4.sh                                   # -> runs/v4/f{0..4}/, runs/v4_b/ (+ ablation v4_np)
```

### Étape 7 — agent A : deuxième échelle (`dense/job_long.sh`, `dense/job_cv_long.sh`)

```bash
cd $T/agentA
sbatch job_long.sh            # pairs_long/ (--with_full --holdout 24) ; runs/long (A1+A2+A3, --scale2) ; runs/long_w (témoin)
sbatch --export=ALL,SRC=long,RUN=v6 job_cv_long.sh     # -> runs/v6 (recette v3) et runs/v6p (recette v4)
```
m5 utilise aussi `v7`, `v7p` et `v8`, produits par le même `job_cv_long.sh` avec d'autres sources (`SRC_PS`, `SRC_TM`) :
les valeurs exactes de ces deux lancements ne sont pas versionnées.

### Étape 8 — k5 : paires avec cadre, bases et affinages (`dense/k5/`)

```bash
cd $T/dense/k5
sbatch job_k5.sh              # run_k5.sh :
                              #   gen_pairs.py --method <m> --reps 2 --rects 2 --with_full --holdout 24 --frame 0.84 --frame_max 1.18 --out pairs_frame
                              #   base k5b PixelSeal / TrustMark (départ v2, 24 000 pas max, budget BUD min) puis affinages en CV (k5, k5p, k5c, …)
# chaînes ajoutées dans un job en cours (srun --overlap) :
bash extra.sh ssb 2           # StegaStamp : base k5b sur pairs_frame (départ v2), puis affinage en CV k5
bash extra.sh a3b 0 pixelseal # base deuxième échelle + cadre k6b (départ agentA/runs/long), puis affinage en CV k6
bash extra.sh a3b 1 trustmark
bash extra.sh a3f 2 pixelseal # mêmes plis k6, dans un autre ordre, pour paralléliser sur un autre GPU
```
`job_k6.sh` appelle un `run_k6.sh` qui n'est pas versionné (il enchaînait les chaînes `extra.sh` ci-dessus).

### Étape 9 — vues par plis et assemblage m5

```bash
python mk_folds.py runs/k5b k5b                   # modèles SANS plis -> mix/k5b/f{0..4}/ (sinon poids train ≠ test)
python mk_folds.py runs/k6b k6b
bash view.sh $T/agentA/runs/v3 stegastamp         # vues à une seule méthode (mix/v_*_<méthode>/f*) quand un run en contient plusieurs
bash asm.sh m5 <liste des dossiers f0..f4 des runs ci-dessous>   # -> out/sub_m5.pt, cv/cv_m5.npz, comparaison à la n° 2170
bash sub.sh out/sub_m5.pt "k5 m5 : …"             # soumission (clé d'équipe lue hors git), journal dans soumissions.csv
```
Composition de m5 (note de soumission, `dense/k5/soumissions.csv`) :
« 2e échelle + cadre (**k6 ×2, base k6b ×2**) + **base k5b** + modèles 2e échelle agent A (**v6, v6p, v7, v7p, v8**) ;
StegaStamp **v3 + v4 + base cadre k5b ×2 + k5 ×2** ». « ×2 » se lit comme un dossier passé deux fois à `--runs`, ce
qui double son poids dans la moyenne (lecture probable, la liste exacte de la commande est restée sur JURECA).
`asm.sh` fixe le reste : 75 rectangles TrustMark, 8 filtres guidés (`none c:8:0.001 c:8:0.01 c:16:0.01 c:16:0.001
c:16:0.0001 c:24:0.001 c:8:0.0001`) choisis hors pli.

## 5. Ce qui n'a pas marché (mesuré, à ne pas refaire)

- Lissage gaussien des logits à la place du filtre guidé (B1) : +0,0000. Règle « rectangle ajusté » `orect` (B3) :
  −0,0002, elle ne se déclenche jamais sur les vraies cartes. Calage des probabilités avant top-k (C11) : −0,0016.
- Graines supplémentaires de la recette v4 (v4_c, v4_d) : −0,0017. Deuxième tour de pseudo-étiquettes (v5) : −0,0007.
- Mélange avec les têtes R5 sur features internes : −0,0006. Régime lu sur la carte : +0,000.
- Rectangles StegaStamp par relecture du message (B9) : −0,0002, la carte dense est déjà aussi précise.
- Message en entrée (corrélation du motif recalculé, A6) : AUC 0,48-0,51 sur les vraies images. Essai B (rejouer le
  tatoueur) : AUC ≤ 0,74 même en synthétique. Filtre adapté local avec le message : AUC ≤ 0,73.
- Segmenteur U-Net RGB : apprend seulement le régime. Propositions SAM seules : +0,027, SAM ne retrouve pas les trous.
- Cadre PixelSeal par recherche exacte : non concluant. Mise au cadre des vraies images avant le détecteur : +0,013 sur
  TrustMark full seule, absorbée par l'augmentation de cadre. Pseudo-étiquettes sur les nouveaux modèles, vues
  multiples à l'inférence : pas de gain.
- Validé en CV mais **pas dans m5** : recalage des zones non-full sur des segments SAM (+0,009), spécialiste « trous »
  TrustMark (full 0,696 contre 0,678).

## 6. Fichiers

| Fichier | Rôle |
|---|---|
| `starter_kit/` | kit officiel : tatoueurs et décodeurs (`wmloc/watermarks.py`), chargement des données, format de soumission |
| `t1_metric.py`, `test_t1_metric.py` | métrique officielle (mIoU par image, sous-échantillonnage 512 → 128), plis stratifiés, top-k ; test |
| `compute_signals.py` | log-vraisemblance globale du message par image → régime full / non-full |
| `rect_search.py` | recherche du rectangle tatoué par décodage de recadrages (rectangles TrustMark du test) |
| `sam/sam_feats.py`, `sam/job_sam.sh` | segments SAM des images (formes d'objets pour les composites) |
| `dense/common.py` | chemins, réserve d'images, `MaskSampler` (formes et statistiques réelles des trous), modèle `DenseDet` |
| `dense/gen_pairs.py` | paires (x, embed(x, m)) sans cadre (v2, agent A) |
| `dense/train_dense.py` | entraînement / affinage en CV / pseudo-étiquettes / deuxième échelle (version agent A) |
| `dense/assemble.py`, `dense/compare_cv.py` | assemblage, règles choisies hors pli, filtre guidé, rectangles ; comparaison CV appariée |
| `dense/submit_file.py` | envoi d'un `.pt` au serveur (clé `WMLOC_API_KEY`), journal `soumissions.csv` (via `../../src/submit.py`) |
| `dense/job_dense.sh`, `job_cv.sh`, `job_v4.sh` | v2, v3, v4 |
| `dense/job_long.sh`, `job_cv_long.sh` | agent A : entraînement long deuxième échelle, puis affinage en CV (v6, v6p, …) |
| `dense/k5/gen_pairs.py` | paires **avec cadre** (`--frame`) |
| `dense/k5/train_dense.py` | `train_dense.py` + mise au cadre optionnelle des vraies images (`--frames`), `--predict_only` |
| `dense/k5/run_k5.sh`, `job_k5.sh`, `extra.sh`, `job_k6.sh` | lanceurs k5b, k5, k6b, k6, StegaStamp avec cadre |
| `dense/k5/mk_folds.py`, `view.sh` | vue par plis d'un modèle sans plis ; vue à une seule méthode |
| `dense/k5/asm.sh`, `sub.sh` | assemblage + comparaison à la n° 2170 ; soumission |
| `dense/k5/{common,assemble,compare_cv,submit_file}.py` | liens symboliques vers `../` ajoutés pour cette branche : `asm.sh` et les scripts k5 les appellent depuis `k5/` |
| `dense/soumissions.csv`, `dense/k5/soumissions.csv` | journaux des soumissions (scores, sha1, notes) |

## 7. Limites de cette reconstitution

- Les poids, probabilités (`runs/`), paires et fichiers soumis sont sur JURECA, pas dans git.
- Non versionnés, donc **absents** : la liste exacte des dossiers passés à `asm.sh m5`, les réglages de `v7`/`v8`,
  `run_k6.sh`, la commande du job `rect_search` du test, la provenance de `convnext_tiny.safetensors`.
- `common.py` a changé dans la nuit (A1a : trous tirés de la distribution réelle). Relancer v2 à v4 avec ce code
  donne des variantes proches, pas des clones bit à bit (même remarque dans `job_seeds.sh`, branche `main`).
- `dense/k5/frames.py` et les autres scripts de diagnostic de `k5/` (geo*, diag*, sens, canon_eval) ont servi à
  **trouver** le cadre, pas à produire m5 : ils sont restés sur la branche `task1-k5`.
