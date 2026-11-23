# Cours : notre méthode pour la tâche 2 (Model Equality), expliquée et critiquée

> Écrit le 22/11 vers 10 h 45, pendant l'épreuve, à partir des journaux de travail de la tâche 2
> (étapes 1 à 19, et résultats de l'analyse boîte blanche).
> But : comprendre ce qu'on a fait, distinguer ce qui est une méthode de ce qui vient du
> leaderboard, et savoir l'expliquer. Les calculs n'ont pas été relancés : les chiffres sont ceux
> des journaux. Les chemins `k/…` et `best/…` désignent le dépôt de travail de l'équipe ; leurs
> équivalents ici sont dans [`task2/src/structure/`](../../task2/src/structure/) et
> [`task2/results/`](../../task2/results/).

## 1. Le problème en une phrase

33 endpoints d'API servent des LLM. Il faut dire lesquels sont **le même modèle** (mêmes poids),
en ignorant les déguisements **cosmétiques** (prompt système, persona, watermark, température) et en
repérant les **changements de poids** (fp8, LoRA à différentes forces). Score = 0,1 × MCC_famille +
0,9 × MCC_modèle, calculé sur des **paires** d'endpoints. Budget : 100 réponses par endpoint, pour
toute la compétition.

C'est un test d'hypothèses en cascade : pour chaque paire, H₀ « mêmes poids ».

## 2. La méthode, étape par étape

### Étape 1 — Les familles (MCC_famille = 1,00)

Les endpoints disent eux-mêmes qui ils sont (« I'm Llama », « I'm Qwen ») et réussissent ou ratent
les mêmes tâches simples. Trois familles de 11 : LLAMA, QWEN_A, QWEN_B. C'est facile, et ça ne
compte que pour 10 % du score.

### Étape 2 — Le piège découvert : le cosmétique domine

Sur les endpoints d'entraînement `ref` (vérité connue), on a mesuré qu'un prompt système éloigne
plus un endpoint de sa base (distance 0,730) qu'une quantification fp8 (0,628). **Conséquence** :
une distance « boîte noire » naïve regroupe les déguisés ensemble et la base avec ses variantes de
poids. C'est exactement l'inverse de ce qu'on veut. Premier essai : 0,05.

**Exemple pour comprendre** : un même modèle à qui on dit « tu es un pirate » répond très
différemment de lui-même sans consigne, alors que sa version fp8 répond presque comme lui. La
distance mesure la forme des réponses, pas les poids.

### Étape 3 — Identifier le modèle exact de chaque famille (boîte blanche)

On calcule, pour chaque réponse déjà payée, sa probabilité sous des modèles candidats téléchargés
(teacher forcing : on fait « relire » la réponse au modèle local, token par token). Le bon
candidat est celui sous lequel les réponses sont les plus probables.

| Famille | Modèle retenu | NLL/token | Concurrent |
|---|---|---|---|
| QWEN_B | Qwen2.5-1.5B-Instruct | 0,39 | 0.5B : 0,82 |
| QWEN_A | Qwen2.5-3B-Instruct | 0,35 | 1.5B : 0,74 |
| LLAMA | Llama-3.2-1B-Instruct | 0,63 | 3B : nettement pire |
| `ref` | SmolLM2-360M-Instruct | 0,94 | 1.7B : 1,41 |

**Pourquoi ce n'est pas du « brute force »** : on choisit parmi 2 à 4 candidats, avec des écarts
énormes, et **sans aucune étiquette ni score**. C'est une sélection de modèle par maximum de
vraisemblance. Le papier MET lui-même suppose qu'on dispose du modèle de référence. Et le
classement final porte sur les **mêmes 33 endpoints** (paires cachées), donc rien ne doit se
généraliser à d'autres modèles.

### Étape 4 — Un test par type de variante, validé sur les `ref`

Une fois la base connue, on a un étalon commun pour les 11 endpoints d'une famille. Chaque test a
été validé là où la vérité est connue avant d'être appliqué :

| Variante | Test | Sur les `ref` | Idée statistique |
|---|---|---|---|
| fp8 | rapport de vraisemblance Σ[log p_fp8(y) − log p_base(y)], fp8 simulé localement | ref-02 : +19 ; les 8 autres : −41 à −225 | **Neyman-Pearson** : quand on connaît les deux hypothèses, le rapport de vraisemblance est le test le plus puissant |
| T = 1 | température qui maximise la vraisemblance des réponses | ref-08 : 1,004 ; plain : 0,715 | estimation par maximum de vraisemblance d'un paramètre de décodage |
| LoRA | corrélation des résidus de fréquence de tokens (observé − attendu sous la base) | les 3 LoRA corrélés entre eux (0,31 à 0,75), le reste < 0,2 | les trois LoRA sont le même adaptateur à trois forces → même direction d'écart |
| watermark | taux de réponses répétées à l'identique | ref-06 : 37 % contre 5 à 15 % | signature empirique : le watermark rend la sortie plus « collante » |
| prompt système / persona | faire réciter la consigne | ep-08 récite un prompt système ; ep-16, ep-20, ep-23 une persona | preuve directe |

**Le principe** : d'abord **reconnaître et neutraliser le cosmétique** (preuve par endpoint),
ensuite seulement chercher les changements de poids. Ce qui reste indétectable (LoRA 0,3, parfois
le fp8 sur le gros modèle) est **absorbé** dans le groupe de base plutôt que deviné : se tromper en
isolant un endpoint coûte plus de paires fausses que le laisser dans le groupe.

### Étape 5 — Construire la partition

Par famille : un groupe de base (les cosmétiques + la base), le reste isolé. Hypothèse « 6 + 5 »
(6 cosmétiques, 5 variantes de poids), cohérente avec les `ref` et avec les scores observés.

## 3. Ce qui n'est PAS de la méthode : l'usage du leaderboard

Il faut le savoir et le dire honnêtement. Entre 1 h et 5 h (étapes 14 à 18), l'équipe a :

1. **soumis des diagnostics** : deux familles en « tout différent », la troisième découpée, pour
   mesurer famille par famille où l'on perdait (1 h 19, 1 h 50, 2 h 21) ;
2. **énuméré les structures compatibles** avec les 7 scores publics obtenus (`k/fit.py`) ;
3. **choisi la soumission suivante par un calcul bayésien** à partir de ces scores
   (`k/bayes.py`), avec une **boucle de soumission automatique** (`k/loop.py`).

C'est se servir du leaderboard comme d'un **oracle**. C'est permis, mais l'énoncé avertit :
« a submission that is tuned to the leaderboard can score high there and close to nothing in the
final ranking ».

## 4. Quelle part du 0,85 vient des preuves, quelle part du leaderboard ?

Partition à 0,85 (`best/BEST.csv`, soumission n° 3549). Lecture endpoint par endpoint d'après
[`task2/README.md`](../../task2/README.md) § 2 et [`task2/results/endpoint_evidence.csv`](../../task2/results/endpoint_evidence.csv), plus
précis que les journaux :

| Famille | Groupe de base | Placé par une preuve validée sur les `ref` | Placé par les scores publics | Jugement (aucune preuve décisive) |
|---|---|---|---|---|
| QWEN_A | 01, 02, 04, 12, 23, 33 | 02 (prompt système), 23 (persona), 04 (watermark, répétitions 0,36), 33 (T = 1) | découpage boîte noire de 22 h 45, **confirmé par le diagnostic de 1 h 19** (0,55, valeur attendue d'un découpage juste) | — |
| QWEN_A | isolés : 14, 17, 18, 24, 25 | 17, 18, 25 (résidus corrélés, MMD par permutation) | idem (découpage confirmé) | — |
| QWEN_B | 03, 06, 07, 08, 20, 21, 30 | 08 (prompt système), 20 (persona), 06 (watermark, 0,25), 30 (T = 1) | **03, 07** (P = 1,00) | **21** (un seul de {21, 32} est dans le groupe, P = 0,50 chacun) |
| QWEN_B | isolés : 22, 26, 28, 32 | 22 (fp8, rapport de vraisemblance), 28 (MMD par permutation, fort décalage) | **26** (P = 0,31) | **32** |
| LLAMA | 10, 13, 16, 27, 29, 31 | 29 (prompt système : préambule verbeux), 16 (persona), 10 (watermark, 0,20), 31 (T = 1) | **13** (P = 0,62) | **27** (P = 0,48) |
| LLAMA | isolés : 05, 09, 11, 15, 19 | 05 et 11 (résidus corrélés), 15 (fp8), 09 (axe dose-effet du LoRA) | — | **19** (P = 0,57) |

**Bilan** : 25 endpoints sur 33 sont placés par une preuve ou par un découpage confirmé, **4** par
les scores publics (03, 07, 13, 26) et **4** au jugement (21, 32, 27, 19). [`task2/README.md`](../../task2/README.md)
le dit aussi.

## 5. Un point à surveiller : ep-28 et ep-29

L'analyse boîte blanche sur CPU (journal de l'analyse boîte blanche, § C) lisait ces deux
endpoints autrement que la partition finale :

| Endpoint | Analyse CPU | Partition à 0,85 | Lecture |
|---|---|---|---|
| ep-28 (QWEN_B) | signature T = 1 → cosmétique, groupe de base | **isolé** (fort décalage au test MMD) | désaccord réel : le test MMD voit les variantes fortes, mais une T = 1 décale aussi les réponses |
| ep-29 (LLAMA) | T = 1,0 et 27 % de réponses tronquées → signature LoRA 1,5 | **groupe de base** (prompt système verbeux) | probablement compatible : un préambule verbeux allonge aussi les réponses, ce que l'analyse CPU a pu prendre pour un LoRA |

## 6. Risques pour le classement final

1. **Les paires cachées.** Le score final porte sur les autres paires des mêmes endpoints. Une
   preuve correcte sur un endpoint améliore les paires publiques **et** cachées. Un placement deviné
   grâce au score public peut être juste… ou juste chanceux sur la moitié publique.
2. **La malédiction du gagnant.** Après une vingtaine de soumissions qui ne diffèrent que de
   quelques endpoints, la meilleure est en partie la plus chanceuse. Un endpoint déplacé ne touche
   qu'environ 3 paires publiques, et le score revient arrondi. Il faut s'attendre à un score final
   un peu inférieur à 0,85.
3. **L'hypothèse 6 + 5** n'est pas prouvée pour chaque famille : elle structure les choix au
   jugement (19, 21, 27, 32).

## 7. En résumé

- **Le cœur** : « Une distance boîte noire confond cosmétique et poids. Nous avons identifié le
  modèle de base de chaque famille par vraisemblance, puis appliqué un test par type de variante,
  chacun validé sur les endpoints de référence. Le fp8, jugé indétectable en boîte noire, se voit
  avec un rapport de vraisemblance contre une version fp8 simulée. »
- **L'honnêteté** : « Pour une poignée d'endpoints que les données ne départagent pas, nous avons
  utilisé les scores publics. Nous savons lesquels, et c'est la partie la moins solide. »
- **L'angle statisticien** : Neyman-Pearson pour le fp8, maximum de vraisemblance pour la
  température, validation sur une vérité connue, et le choix d'« absorber » plutôt que deviner parce
  qu'une erreur de regroupement coûte moins de paires qu'une erreur d'isolement.

### Questions fréquentes

| Question | Réponse courte |
|---|---|
| « Identifier le modèle, ce n'est pas tricher ? » | Non : ni étiquette ni score, 2 à 4 candidats, écarts énormes. MET suppose un modèle de référence. |
| « Et si le serveur avait quantifié autrement ? » | C'est le risque principal du test fp8 ; il a été validé sur ref-02, qui passe par le même serveur. |
| « Pourquoi pas la MMD du papier ? » | Essayée : elle voit les variantes fortes, pas les légères, et elle réagit au cosmétique. |
| « Combien de requêtes ? » | Environ 3 256 sur 3 300 ; la boîte blanche n'en coûte aucune, elle relit les réponses déjà payées. |

Voir aussi : [notes de lecture MET](../papers/model_equality_testing.md),
[cours MET](model_equality_testing_course_fr.pdf), [énoncé de la tâche 2](../tasks/task2_model_equality.md).
