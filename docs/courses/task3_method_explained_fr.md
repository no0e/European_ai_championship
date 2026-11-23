# Cours : notre méthode pour la tâche 3 (Unlearning), expliquée et critiquée

> Écrit le 22/11 vers 10 h 50, pendant l'épreuve, à partir du journal des 92 soumissions de la
> tâche 3 et de [`task3/README.md`](../../task3/README.md). Même plan que
> [`task2_method_explained_fr.md`](task2_method_explained_fr.md) : comprendre ce qu'on a fait,
> séparer la méthode de ce qui vient du leaderboard, savoir l'expliquer. Les calculs n'ont pas été
> relancés. Les chemins `src/…` renvoient à `task3/src/` ; ceux en `k3/…` désignent le dépôt de
> travail de l'équipe (simulateur, diagnostics), non inclus ici.

## 1. Le problème en une phrase

On reçoit un ViT-B/16 (100 classes, entraîné de zéro) et 3 500 images **sans étiquettes** à lui
faire oublier. Il faut rendre des poids qui se comportent comme **gold** : la moyenne de plusieurs
modèles réentraînés sans ces images. Pas de jeu « à garder », pas de jeu de validation.

Le score multiplie deux termes :
- **forget** = `mean_std_score × prediction_score`, sur les images à oublier **et** sur des copies
  altérées cachées (moitié-moitié) :
  - `mean_std_score` : la moyenne **et l'écart-type** de la perte par image doivent coller à ceux de
    gold ;
  - `prediction_score` : image par image, la distribution prédite (softmax à T = 2) doit coller à
    celle de gold (KL) ;
- **retain** = précision × similarité KL avec orig sur un jeu de contrôle caché ;
- et **0** si la précision cachée tombe sous 70 %.

## 2. L'idée clé : oublier, ce n'est pas se tromper

**L'erreur naïve** : faire monter la perte sur les images à oublier (montée de gradient, SCRUB
poussé fort). Le modèle se trompe, mais **trop et de façon trop uniforme**. Le papier SCRUB le
dit à sa façon (§ 3.2, SCRUB+R) : une erreur anormalement haute sur les exemples oubliés les rend
identifiables. C'est un effet Streisand : un exemple oublié « trop fort » se remarque.

**Ce que fait gold** : il n'a jamais vu ces images, donc pour lui ce sont des **images de test**.
Il en classe certaines juste, d'autres faux, avec des pertes **fortes et très dispersées** (de
l'ordre de 5 ± 4,5 d'après un modèle du score ajusté sur sept soumissions, cf. § 5). Les images
faciles restent justes, les difficiles deviennent fausses.

**Exemple** : une photo de bateau très typique, gold la reconnaît sans l'avoir vue. Une photo de
lanterne ambiguë, gold la prend pour une figurine. Le modèle d'origine, lui, a « appris par cœur »
les deux. Bien oublier, c'est rendre la première **un peu moins sûre** et la seconde **fausse**, pas
rendre les deux fausses.

## 3. La méthode, étape par étape

### 3.1 Un simulateur, pour avoir une vérité terrain

Sur la vraie tâche, on ne voit gold qu'à travers un score public. Pour comprendre le problème sans
gaspiller de soumissions, l'équipe a construit un **simulateur** (`k3/sim/`) : un ViT entraîné de
zéro sur CIFAR-100, la même structure de jeu à oublier (45 classes, mêmes effectifs), **4 golds
réentraînés** et le barème exact.

C'est le même rôle que les endpoints `ref` de la tâche 2 : un laboratoire où l'on connaît la
réponse. Il a donné en une heure la **structure** de la méthode :
- adoucir les sorties ne vaut que 0,04 d'oubli (comme sur le leaderboard) ;
- fixer la marge de la classe d'origine selon la difficulté vaut 0,37 à 0,46 ;
- la distillation KL seule ne reproduit pas la queue des pertes.

Ses **réglages fins**, eux, ne se sont pas transportés (cf. § 4).

### 3.2 Construire une cible pour chaque image (`src/tg.py`)

1. **Classe d'origine** y = prédiction d'orig. Les étiquettes ne sont pas fournies ; orig a été
   entraîné sur ces images, donc sa prédiction est très probablement la bonne étiquette.
2. **Difficulté** u(x) ∈ [0, 1] : le **rang** de l'image selon la marge d'orig, moyennée sur l'image
   propre et 4 vues altérées (0 = facile, 1 = difficile).
3. **Marge imposée** à la classe d'origine (son logit moins le meilleur des autres) :

   m(x) = μ − σ · Φ⁻¹(u(x)),   avec μ = −2,5 et σ = 7.

   **C'est une transformation par quantiles** : on range les images par difficulté, puis on leur
   attribue les quantiles d'une loi normale N(μ, σ²). Les marges imposées suivent donc exactement
   cette loi, et l'ordre de difficulté est respecté.

   **Exemple chiffré** :
   - image médiane (u = 0,5) : Φ⁻¹ = 0 → m = −2,5, légèrement fausse ;
   - image difficile (u = 0,9) : Φ⁻¹ ≈ 1,28 → m ≈ −11,5, nettement fausse ;
   - image facile (u = 0,1) : Φ⁻¹ ≈ −1,28 → m ≈ +6,5, encore juste (puis divisée par 2 par
     `--pos_scale 0.5` : les images que gold garde justes sont moins sûres d'elles).

   **Conséquence implicite** : P(m > 0) = Φ(μ/σ) = Φ(−0,36) ≈ 0,36. Ces réglages font donc comme
   si gold classait juste environ **36 %** des images oubliées, comme une précision de test.
4. **Les autres logits** sont ceux d'orig, légèrement écartés (`tb = 1,4`). Seul le logit de la
   classe d'origine bouge : on laisse le modèle choisir lui-même vers quelle mauvaise classe aller,
   ce qui ressemble à ce que ferait gold.

### 3.3 Distiller vers ces cibles (`src/k3.py distill`)

L'élève part d'orig (30 époques, lr 1e-4) et apprend :
- **KL à T = 2 vers les cibles** sur les images à oublier → vise le `prediction_score` ;
- **une perte Huber sur la cross-entropy par image** → vise le `mean_std_score`. Sans elle, la
  queue de la distribution des pertes n'est pas reproduite, donc l'écart-type est faux ;
- **des copies altérées** (recadrage, miroir, couleur, flou, bruit) avec une marge abaissée de 4,5
  de plus. Diagnostic mesuré (`k3/view_stats.py`) : l'élève naïf oubliait **moins** sur les copies
  (perte 3,18 sur un recadrage, 4,01 sur un miroir, contre 4,58 sur l'image propre), alors que gold
  perd plutôt plus. Corrigé : 0,3524 → 0,381 ;
- **5 000 images COCO en ancres**, sur lesquelles l'élève doit rester égal à orig → vise le
  `retain` (similarité KL avec orig).

### 3.4 Moyenner les poids de six modèles (`src/k3.py avg`)

Six modèles du même plateau de réglages, tous partis d'orig, moyennés paramètre par paramètre :
**0,4700**, contre 0,459 à 0,465 pour chacun seul.

**Pourquoi ça marche** : chaque entraînement a un bruit propre (±0,006, mesuré en relançant un
réglage avec une autre graine). La moyenne retire ce bruit. C'est aussi cohérent avec la cible :
**gold est lui-même une moyenne de plusieurs modèles réentraînés**, donc une sortie lissée.
(Moyenner des poids n'a de sens que parce que les six modèles partent du même point et en restent
proches ; avec douze, 0,4690 : pas de gain supplémentaire.)

## 4. Ce qui est méthode, ce qui vient du leaderboard

| Élément | Origine | Solidité |
|---|---|---|
| « Oublier = se comporter en test », marge par rang | simulateur + lecture du barème | **méthode** |
| Perte Huber sur la CE par image | simulateur (queue des pertes) | **méthode** |
| Marge abaissée sur les copies altérées | diagnostic local mesuré (`view_stats.py`) | **méthode** |
| Ancres COCO | lecture du barème (terme retain) | **méthode** |
| Moyenne de poids | bruit d'entraînement mesuré | **méthode** |
| Valeurs μ = −2,5, σ = 7, tb = 1,4, `view_delta` 4,5, `pos_scale` 0,5, 30 époques, 35 % de vues fortes | **recherche axe par axe sur le score public** | réglé sur le leaderboard |
| « Gold perd vers 5,2 ± 4,5 » | modèle du score ajusté sur 7 soumissions | **déduit du leaderboard** |

**Le point honnête** : la structure vient d'un raisonnement et d'un simulateur ; les **réglages
fins** viennent du leaderboard. Le simulateur indiquait même l'inverse pour `tb` (resserrer les
autres logits), et le leaderboard l'a contredit (tb 1,0 : 0,4307 ; 1,4 : 0,4484). Au total, 92
soumissions.

## 5. Est-ce du surapprentissage du leaderboard ?

**Moins que pour la tâche 2, et voici pourquoi.**

- **Ce qu'on règle est global.** Une dizaine de paramètres qui s'appliquent à toutes les images,
  pas une décision image par image. Peu de degrés de liberté, donc peu de place pour coller au
  hasard de la partie publique.
- **Public et final sont tirés de la même population.** L'énoncé : environ 30 % des données de
  scoring pour le public, 70 % pour le classement final, partagés par un hachage de l'identifiant
  de chaque image (une copie altérée suit son image source). Un bon réglage global sur 30 % est un
  bon réglage sur 70 %.
- **Comparaison avec la tâche 2** : là-bas, le leaderboard servait à décider la place
  d'**endpoints précis**, ce qui ne se vérifie que sur la moitié publique des paires.

**Mais deux risques restent.**

1. **La malédiction du gagnant.** Parmi 92 soumissions, la meilleure est en partie la plus
   chanceuse. Les écarts entre les meilleures (0,4605, 0,4646, 0,4700) sont du même ordre que le
   bruit d'entraînement (±0,006). Le bruit de l'**évaluation** sur 30 % des images n'a pas été
   mesuré et s'y ajoute. Il faut s'attendre à un score final un peu différent, probablement un peu
   inférieur.
2. **La moyenne de poids est le choix le plus robuste** : elle ne dépend pas du modèle le plus
   chanceux, elle donne le niveau réel du réglage. C'est un bon argument pour l'avoir soumise en
   dernier.

## 6. Ce qui n'a pas marché, et ce que ça apprend

| Idée | Résultat | Leçon |
|---|---|---|
| Adoucir toutes les sorties (T = 8) | 0,024 | gold n'est pas « moins sûr partout », il est faux sur certaines images |
| SCRUB, plis extrapolés | 0,15, forte dérive | pousser loin de l'enseignant dépasse la cible (effet Streisand) |
| « Gold n'a jamais vu les 9 classes lourdes » | 0,1382 contre 0,3455 | réfuté : gold connaît toutes les classes |
| Classement par voisins (k4) | +30 % sur le simulateur, −9 % sur le réel | un réglage fin du simulateur ne se transporte pas |
| Garder les vues fortes identiques à orig | 0,4268 | toute protection près des images à oublier empêche l'oubli de s'étendre aux copies, qui comptent pour moitié |
| Même mauvaise classe pour toutes les copies d'une image | 0,3445 | la mauvaise classe doit être lue sur chaque copie |
| 50 époques au lieu de 30 | 0,4027 | mieux coller aux cibles fait dériver le reste : le retain coûte plus que l'ajustement ne rapporte |

## 7. En résumé

- **Le cœur** : « Oublier, ce n'est pas se tromper : c'est se comporter comme un modèle qui voit ces
  images pour la première fois. On fixe pour chaque image une marge cible par transformation de
  quantiles de son rang de difficulté, puis on distille, en visant à la fois la distribution
  prédite et la distribution des pertes, sur les images et leurs copies altérées. »
- **L'angle statisticien** : transformation par quantiles, un terme de perte pour chaque moment
  mesuré par le barème (moyenne et écart-type), moyenne de modèles pour réduire la variance,
  simulateur pour avoir une vérité terrain.
- **L'honnêteté** : « La structure vient du simulateur ; les réglages fins viennent du score
  public, ce qui est un risque limité parce qu'ils sont globaux et que public et final sont tirés
  de la même population. »

### Questions fréquentes

| Question | Réponse courte |
|---|---|
| « Pourquoi pas SCRUB, le papier recommandé ? » | Essayé : 0,15. Il éloigne l'élève de l'enseignant sans cible précise ; le barème demande de **reproduire** gold, pas de s'éloigner d'orig. |
| « Sans étiquettes, comment savez-vous la classe ? » | Prédiction d'orig, qui a été entraîné sur ces images, donc très probablement juste. |
| « Une garantie théorique d'oubli ? » | Aucune ; c'est une imitation de gold mesurée par le barème, comme dans SCRUB. La piste « certified removal » a été testée : sans effet ici. |
| « Et la barrière des 70 % ? » | Loin : neutraliser 3 classes entières ne coûte que 5 % du score. |
| « Ça marcherait sur un autre modèle ? » | La structure oui (validée sur un simulateur différent) ; les réglages fins seraient à refaire. |

Voir aussi : [notes de lecture SCRUB](../papers/scrub_unlearning.md),
[cours SCRUB](scrub_unlearning_course_fr.pdf), [énoncé de la tâche 3](../tasks/task3_unlearning.md).
