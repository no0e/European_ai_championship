# Courses and explanations (in French)

Material the team wrote to learn the three papers before the event and to explain our methods during it. It is in
French, the team's working language; the English summaries are the [reading notes](../papers/README.md) and the task
READMEs.

## Beginner's courses on the three papers (written before the event)

Each course starts from zero (networks, losses, probabilities), then goes through the paper section by section,
equation by equation, and ends with the official code, experiments we ran ourselves, and the limits of the method.

| Course | Paper | Pages |
|---|---|---|
| [pixel_seal_course_fr.pdf](pixel_seal_course_fr.pdf) | Pixel Seal (task 1) | 46 |
| [model_equality_testing_course_fr.pdf](model_equality_testing_course_fr.pdf) | Model Equality Testing (task 2) | 44 |
| [scrub_unlearning_course_fr.pdf](scrub_unlearning_course_fr.pdf) | SCRUB (task 3) | 43 |

Figures marked "[article]" in the courses are reproduced from the original papers for study purposes; figures marked
"[TP]" come from our own runs.

## Our methods, explained and criticised (written during the event)

| Note | Content |
|---|---|
| [task2_method_explained_fr.md](task2_method_explained_fr.md) | Task 2: the method step by step, the statistical idea behind each test, which endpoints are placed by evidence and which by public scores, risks for the final ranking |
| [task3_method_explained_fr.md](task3_method_explained_fr.md) | Task 3: why forgetting is not being wrong, the rank-based quantile targets with a worked example, what is method and what was tuned on the leaderboard |

Task 1 is explained in English in [`../../task1/README.md`](../../task1/README.md).

## Toy scripts (CPU, a minute each)

| Script | What it shows |
|---|---|
| [toy/met_toy.py](toy/met_toy.py) | MMD with the Hamming kernel and its permutation p-value, on synthetic token sequences (numpy only) |
| [toy/scrub_minimal.py](toy/scrub_minimal.py) | SCRUB and SCRUB+R on a small synthetic dataset, against a retrained oracle, with a basic membership-inference attack (torch, scikit-learn) |

```bash
python docs/courses/toy/met_toy.py
python docs/courses/toy/scrub_minimal.py
```
