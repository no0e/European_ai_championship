# Tâche 3 : unlearning (faire oublier 3 500 images à un ViT)

**Meilleur score public : 0,4700011591814447** (affiché 0,4700), 22/11 à 07 h 59.
Fichier soumis : `k3/sub/plat6.pt`, **moyenne de poids de six modèles** de la méthode « k3 »
(ps1, ps10, st1, bt1, bt2, y1, notés seuls de 0,4591 à 0,4646).

Source : branche `task3-k3` (commit `1f4b369`). Journal complet : `task3/history/HISTORY.md` (92 soumissions de
l'équipe, statistiques locales de chaque candidat) et `task3/history/narrative.md` sur cette branche ; journaux bruts
des runs dans `task3/history/logs/`.

---

## 1. Le problème

- Modèle : **ViT-B/16** classant **100 classes**, fourni en `model.pt` (« orig ») ; **3 500 images à oublier**
  (`forget_images.npz`), **sans labels**. Aucun jeu à retenir, aucun jeu de test fourni.
- On rend le `state_dict` du modèle désappris. Le serveur le compare à « gold » (moyenne de modèles réentraînés sans
  les images à oublier) :
  - `forget = mean_std_score × prediction_score` : la moyenne et l'écart-type de la perte sur les images à oublier
    doivent ressembler à ceux de gold, et les sorties à T = 2 aussi, image par image (KL(gold‖sub) / KL(gold‖orig)) ;
  - `retain = accuracy_score × kl_similarity` sur un jeu de contrôle caché : rester collé à orig ailleurs ;
  - score = forget × retain, et **0 si la précision sur le jeu caché passe sous 70 %**.
- Une soumission environ toutes les 8 à 10 minutes par équipe.
- Rendre orig tel quel donne 0 (forget = 0). Il ne faut pas « oublier au maximum » mais **imiter un modèle qui n'a
  jamais vu ces images**.

## 2. La méthode finale (k3 + moyenne de poids)

En trois phrases : gold se comporte sur les images à oublier comme un modèle « en test » (souvent faux, pertes fortes
et dispersées). On fabrique donc, pour chaque image, une **cible** = les logits d'orig dont on abaisse seulement le
logit de la classe d'origine, d'autant plus que l'image est « difficile ». Puis on **distille** : l'élève part d'orig,
apprend ces cibles sur les images à oublier et leurs copies altérées, et reste collé à orig sur des images d'ancrage
(COCO).

### 2.1 Les cibles « marge par rang » (`k3/tg.py`)

- Classe d'origine y = prédiction d'orig. Difficulté u(x) ∈ [0, 1] = rang de la marge moyenne d'orig sur l'image propre
  et 4 vues altérées (`--rank mv`).
- Marge imposée à la classe d'origine : `m(x) = mu − sigma · Φ⁻¹(u(x)) − beta · rho_c` (rho_c = poids relatif de la
  classe dans le jeu à oublier) ; les logits des autres classes sont ceux d'orig, multipliés par `tb`.
- Réglage retenu : `mu = −2,5`, `sigma = 7`, `tb = 1,4` (écarter un peu les autres classes), `--pos_scale 0.5`
  (marges positives divisées par deux : les images que gold classe juste sont moins confiantes).

### 2.2 La distillation (`k3/k3.py distill`)

- 30 époques, lr 1e-4, départ = orig, KL à T = 2 vers les cibles + **perte de Huber sur la CE par image**
  (`--w_ce 1`, indispensable : sans elle la queue de pertes n'est pas reproduite).
- Copies altérées des images à oublier (recadrage, miroir, couleurs, flou, bruit) : la moitié du score se joue sur
  elles ; on y abaisse la marge de 4,5 de plus (`--view_delta 4.5`), avec 30 % d'images propres (`--clean_frac`) et
  20 % de vues fortes (`--strong_frac`).
- Ancres COCO val2017 (`anchors.npz`) : l'élève y reste égal à orig (KL).

### 2.3 La moyenne de poids (`k3/k3.py avg`)

Six modèles du même plateau de réglages, tous partis d'orig, moyennés paramètre par paramètre. La moyenne retire le
bruit propre à chaque entraînement (±0,006 mesuré en relançant `v22` avec une autre graine) et fixe le niveau réel du
réglage : **0,4700** contre 0,459-0,465 pour chacun. Moyenne de douze modèles (`plat12`) : 0,4690. C'est aussi le
choix le plus sûr pour le classement final sur les images cachées.

| Modèle | Ce qui le distingue | Public seul |
|---|---|---|
| `ps1` | réglage de référence (`--pos_scale 0.5`) | 0,4605 |
| `ps10` | `ps1`, autre graine (`--seed 1`) | non soumis |
| `st1` | 35 % de vues fortes, 25 % d'images propres | 0,4646 |
| `bt1` | marge dépendant du poids de la classe, `beta = 2`, `mu = −1,1` | 0,4591 |
| `bt2` | idem, `beta = −2`, `mu = −3,9` | 0,4597 |
| `y1` | 30 % de variantes synthétiques (diffusion) traitées comme des copies à oublier | 0,4596 |

## 3. Le raisonnement, étape par étape

| Moment | Meilleur public | Ce qui a changé, et pourquoi |
|---|---|---|
| 21/11 17 h 13 | 0,024 | Adoucissement des sorties (T = 8) + ancres COCO : la cible gold n'est pas « moins confiant partout ». |
| 21/11 après-midi | 0 (×2) | Chirurgie de la tête sur les 9 classes « lourdes » : nuisible. |
| 21/11 22 h 58 | 0,1496 | Plis SCRUB, extrapolés en poids (θ_orig + a·ΣΔ) + température de tête : désapprend peu et dérive beaucoup (accord sur les ancres 0,61). |
| 22 h 00 – 22 h 30 | — | Reprise à zéro. Le modèle a été entraîné de zéro (cosinus ≈ 0 avec 7 checkpoints timm) ; images générées par diffusion, classes fines groupées en familles. |
| 22 h 40 – 23 h 30 | — | **Simulateur** (`k3/sim/` sur la branche source) : ViT entraîné de zéro sur CIFAR-100, même structure du jeu à oublier, 4 golds, barème exact. Il donne la **structure** de la méthode : fixer la marge de la classe d'origine par rang de difficulté (0,37-0,46 d'oubli contre 0,04 pour l'adoucissement) et ajouter la perte sur la CE par image. |
| **23 h 37** | **0,2585** | Première soumission k3 (`r1`). |
| 22/11 00 h 48 | 0,3524 | Calibrage de la marge : optimum vers mu −3, sigma 7. |
| 02 h 33 → 03 h 02 | 0,381 | `k3/view_stats.py` (branche source) : l'élève oubliait **moins** sur les copies altérées (perte 3,18 sur un recadrage, 4,01 sur un miroir, contre 4,58 sur l'image propre) alors qu'un gold perd plutôt plus. Correction `--view_delta` + 70 % de vues. |
| 03 h 21 → 03 h 32 | 0,4484 | Ne plus resserrer les logits des autres classes (`tb 1,0` : 0,4307), puis les écarter (`tb 1,4`). Le simulateur indiquait l'inverse : ses réglages fins ne se transfèrent pas. |
| 03 h 51 → 05 h 56 | — | Vérification axe par axe : optimum local (mu −1,5 : 0,4189 ; −3,5 : 0,3745 ; 50 époques : 0,4027 ; 20 : 0,4289). Bruit mesuré : ±0,006. |
| 06 h 06 | 0,4605 | `--pos_scale 0.5` (`ps1`). 0,3 : 0,4557 ; 0 : 0,4365. |
| 07 h 19 | 0,4646 | 35 % de vues fortes (`st1`), dans le bruit. |
| **07 h 59** | **0,4700** | **Moyenne de poids des six modèles du plateau (`plat6`).** |

Ce qu'on a compris du problème :
1. Gold est « en test » sur les images à oublier : un modèle simple du score, ajusté sur sept soumissions, situe sa
   perte vers 5 ± 4,5.
2. Gold connaît toutes les classes, y compris les 9 classes dont 130 images sont à oublier (`h1` 0,1382 contre 0,3455).
3. La barrière des 70 % est loin : neutraliser 3 classes ne coûte que 5 % du score.
4. Les copies altérées comptent pour moitié, et un élève naïf y oublie moins que sur l'image propre.
5. Toucher au seul logit de la classe d'origine suffit : resserrer les autres nuit, les écarter aide.
6. La rétention est un coût structurel : sur le simulateur, ni vraies images du domaine en ancres, ni négatifs
   adverses, ni sous-ensembles de paramètres ne la remontent de plus de 5 %.

## 4. Pipeline complète, de l'état initial à `plat6.pt`

Tout tourne dans le dossier de l'épreuve (`unlearning/` sur JURECA), qui contient les fichiers officiels :
`model.pt`, `forget_images.npz`, `task_template.py`, et la clé d'équipe dans `.submit_t3.py` (hors git). Disposition :
`unlearning/{task_template.py, model.pt, forget_images.npz, anchors.npz, prepare_anchors.py, k3/, k5/}`.

### Étape 0 — ancres COCO (nœud de connexion, qui a internet)

```bash
# images COCO val2017 (*.jpg) téléchargées dans val2017/
python3 prepare_anchors.py val2017 anchors.npz 5000     # côté court 256, recadrage central 224 -> uint8 [N,224,224,3]
```

### Étape 1 — variantes synthétiques (pour `y1` seulement)

```bash
# diffusers/transformers installés à part : uv pip install --target k3/pylib --no-deps diffusers transformers tokenizers regex
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python k5/gen_variants.py --strength 0.5 --k 2 --shard $i/4 --out k5/var_s0.5_$i.npz & done; wait
python -c "
import numpy as np, glob
z = [np.load(f) for f in sorted(glob.glob('k5/var_s0.5_[0-3].npz'))]
np.savez('k5/syn_s0.5.npz', images=np.concatenate([a['images'] for a in z]), src=np.concatenate([a['src'] for a in z]))"
```
`k3.py` lit `k5/syn_s0.5.npz` (variable `K3_SYN`). La fusion des quatre morceaux n'est pas versionnée : la commande
ci-dessus est **reconstituée** d'après les clés attendues par `k3.py` (`images`, `src`, 2 variantes par image).

### Étape 2 — les six modèles (un GPU chacun, ≈ 6 min ; `k3/rr.sh` = `tg.py` puis `k3.py distill`)

Commandes exactes, relevées dans la file de commandes du job (`task3/history/logs/home_k3q/queue.log` sur la branche source) :
```bash
cd k3
./rr.sh ps1  "--rank mv --mu -2.5 --sigma 7 --tb 1.4 --pos_scale 0.5" "--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --strong_frac 0.2 --clean_frac 0.3 --view_delta 4.5"
./rr.sh ps10 "--rank mv --mu -2.5 --sigma 7 --tb 1.4 --pos_scale 0.5" "--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --strong_frac 0.2 --clean_frac 0.3 --view_delta 4.5 --seed 1"
./rr.sh bt1  "--rank mv --mu -1.1 --sigma 7 --tb 1.4 --pos_scale 0.5 --beta 2"  "--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --strong_frac 0.2 --clean_frac 0.3 --view_delta 4.5"
./rr.sh bt2  "--rank mv --mu -3.9 --sigma 7 --tb 1.4 --pos_scale 0.5 --beta -2" "--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --strong_frac 0.2 --clean_frac 0.3 --view_delta 4.5"
./rr.sh st1  "--rank mv --mu -2.5 --sigma 7 --tb 1.4 --pos_scale 0.5" "--lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 --view_delta 4.5 --clean_frac 0.25 --strong_frac 0.35"
# y1 : pas de ligne dans la file versionnée ; options relevées dans son journal (logs/real_y1.log) :
cp tgt_real/ps1.pt tgt_real/y1.pt        # hypothèse : cibles de ps1 (aucun journal tg.py pour y1)
python k3.py distill --env real --targets k3/tgt_real/y1.pt --lr 1e-4 --w_ce 1 --alphas 1.0 --epochs 30 \
       --strong_frac 0.2 --clean_frac 0.3 --view_delta 4.5 --syn_frac 0.3 --out sub/y1.pt
```
Chaque distillation écrit `sub/<nom>_a1.0.pt` (state_dict) et affiche ses statistiques locales (précision et perte
sur les images à oublier, accord et KL sur les ancres). Sur le GPU d'un nœud de connexion : `K3_AMP=fp16`.

### Étape 3 — moyenne de poids et soumission

```bash
cd ..   # unlearning/
python k3/k3.py avg --env real --inputs "k3/sub/ps1_a1.0.pt,k3/sub/ps10_a1.0.pt,k3/sub/st1_a1.0.pt,k3/sub/bt1_a1.0.pt,k3/sub/bt2_a1.0.pt,k3/sub/y1_a1.0.pt" --out k3/sub/plat6.pt
python k3/submit.py k3/sub/plat6.pt      # attend le délai serveur (« Wait N »), journal output/submit_log.txt
```
Sur le cluster, ces commandes passaient par une file pilotée depuis le home (`k3/home_k3q/job.sh` : une ligne
`<GPU> <commande>` par tâche, un GPU par commande) et une file de soumissions différées (`k3/home_k3q/subw.sh`).

## 5. Ce qui a été écarté (mesuré)

| Idée | Verdict | Mesure |
|---|---|---|
| Adoucissement uniforme des sorties | insuffisant | 0,024 à 0,027 public |
| Chirurgie de la tête des 9 classes lourdes | nuisible | 0,0 (deux essais) |
| Plis SCRUB comme cibles, extrapolés | dépassé | 0,15 ; forte dérive |
| Classement par voisins dans les couches intermédiaires (k4) | ne se transfère pas | gagne sur le simulateur, −9 % sur le réel |
| Gold n'ayant jamais vu les 9 classes lourdes | réfuté | `h1` 0,1382 contre 0,3455 |
| Resserrer les logits des autres classes (tb 0,8) | nuisible | 0,381 contre 0,4307 (tb 1,0) |
| 50 ou 20 époques au lieu de 30 | moins bon | 0,4027 et 0,4289 contre 0,4484 |
| Images justes à marge nulle (`--pos_scale 0`) | moins bon | 0,4365 contre 0,4605 |
| Vues fortes gardées identiques à orig | moins bon | `sk1` 0,4268 |
| Même mauvaise classe pour toutes les copies d'une image | nuisible | `oc1` 0,3445 |
| Pénalité de Fisher (certified removal) | sans effet | simulateur 0,361 contre 0,360 |
| Variantes synthétiques comme ancres du domaine | nuisible | `y2` 0,3995 |
| 50 % de vues fortes | un peu moins bon | `st2` 0,4508 |

Leçon : toute protection placée près des images à oublier (vues fortes gardées, variantes proches en ancres) empêche
l'oubli de s'étendre à leurs versions transformées, qui comptent pour moitié dans le score.

## 6. Fichiers

| Fichier | Rôle |
|---|---|
| `task_template.py`, `requirements.txt` | fichiers officiels : construction du ViT, prétraitement du serveur, chargement des images, format de soumission |
| `prepare_anchors.py` | ancres COCO val2017 → `anchors.npz` |
| `k3/tg.py` | cibles « marge par rang » |
| `k3/k3.py` | environnement réel, vues altérées, distillation (`distill`), moyenne de poids (`avg`) ; contient aussi le mode simulateur et les plis hors-pli, non utilisés ici |
| `k3/rr.sh` | `tg.py` puis `k3.py distill`, avec réessais si le quota de fichiers sature |
| `k3/submit.py` | soumission avec attente du délai serveur et réessais réseau |
| `k3/home_k3q/job.sh`, `k3/home_k3q/subw.sh` | job Slurm piloté par une file dans le home ; file de soumissions |
| `k5/gen_variants.py` | variantes image-vers-image (sd-turbo) des images à oublier, pour `y1` |

## 7. Limites

- Les poids (`k3/sub/*.pt`), les cibles (`k3/tgt_real/`) et les variantes sont sur JURECA, pas dans git.
- Deux étapes ne sont pas versionnées et sont **reconstituées** ci-dessus : la fusion `k5/var_s0.5_*.npz` →
  `k5/syn_s0.5.npz`, et les cibles de `y1` (supposées identiques à celles de `ps1`).
- `k3.py` est la version du commit `1f4b369` : elle a gagné des options après `ps1`, `bt1` et `bt2` (`--ewc`,
  `--syn_frac`, `--llrd`, `--ema`, `--others_src`). Leurs valeurs par défaut (0 ; 1,0 pour `--llrd` ; `view` pour
  `--others_src`) n'activent rien, ce qui redonne le comportement de l'époque.
- Le simulateur (`k3/sim/`) et la piste k4 ont guidé la méthode mais ne produisent rien de `plat6` : ils restent sur
  la branche `task3-k3`.
