# Context — the European AI Championship, Paris stage (21–22 November 2026)

## The event

- **What**: the Paris regional round of the **European AI Championship**, a series of 24-hour competitions on AI and
  security led by the **CISPA Helmholtz Center for Information Security**. The tasks are written by the **SprintML**
  group (Adam Dziedzic, CISPA; Franziska Boenisch, TUM and Fraunhofer AISEC), who also supervised them.
- **Organisers and partners**: CISPA and PoC Innovation, in partnership with Meta, which offered the prize
  (**€10,000** for the first place). About 200 participants, teams of 3 to 4.
- **Talk**: on Saturday, Pierre Fernandez (Meta, co-author of Pixel Seal) gave a talk on invisible image
  watermarking between research and production — post-hoc versus in-model watermarks, what is used in practice,
  robustness and security against adaptive adversaries.
- **Recommended reading** sent before the event: three papers, which turned out to be one per task —
  Pixel Seal, Model Equality Testing and SCRUB. See [papers/](papers/README.md).

## Timeline

| When | What |
|---|---|
| Sat 21/11, 13:00 | Opening, keynote, the three tasks revealed |
| Sat 21/11, 14:30 | Hacking starts |
| Sun 22/11, 12:00 | Hacking ends |
| Sun 22/11, 13:00 | Six-minute presentations by the six best teams, in front of the CISPA technical jury |
| Sun 22/11, 14:30 | Awards |

## The three tasks

| Task | Recommended paper | What is submitted | Score |
|---|---|---|---|
| [1 — Watermark localisation](tasks/task1_watermark_localisation.md) | Pixel Seal | a 128×128 binary mask for each of 1,020 test images | per-image mIoU over the two classes |
| [2 — Model equality testing](tasks/task2_model_equality.md) | Model Equality Testing | a CSV `endpoint,family,model` for 33 LLM endpoints | 0.1 × MCC(family) + 0.9 × MCC(model) over 528 pairs |
| [3 — Unlearning](tasks/task3_unlearning.md) | SCRUB | the `state_dict` of an unlearned ViT-B/16 | forget × retain, 0 below 70 % held-out accuracy |

All three are **auditing / trustworthy-ML problems**: finding *where* a watermark is (so that edits are not
attributed to the watermark owner), finding *which* model an API really serves, and checking that a model has
*really* forgotten data rather than merely being damaged on it.

## Rules that shaped our work

- **Live leaderboard, hidden final ranking.** Each task has its own leaderboard; the public score is computed on a
  part of the test data (about 30 % for tasks 1 and 3, part of the 528 pairs for task 2). The final ranking uses
  the hidden part. All three statements warn against tuning to the public score; task 2 says so in capitals.
- **Budgets.** Task 2: 100 paid answers per endpoint for the whole event, and one submission every 30 minutes.
  Tasks 1 and 3: the servers rate-limit submissions.
- **Compute.** GPUs at the Jülich Supercomputing Centre (JURECA-DC, A100 nodes, Slurm, project `training2666`),
  shared across the team through an official setup script. Access closed with the end of the event.
- **No extra data** for task 3: only the original model and the 3,500 unlabelled images to forget.

## How we worked

- **Preparation** (the week before): reading notes and a beginner's course on each recommended paper, toy
  implementations of the core ideas, and a GPU setup tested on the cluster. The courses are in
  [courses/](courses/README.md).
- **During the 22 hours**: one rule for all three tasks, which is also the title of our slides: **measure before tuning** — a local metric identical to
  the official one (task 1), endpoints with a known answer (task 2's `ref-01`…`ref-09`), a simulator with a known
  gold model (task 3).
- **After the event**: the code of the best submission of each task was reorganised into this repository, with a
  configuration file, numbered steps and a README per task.

## Results (public leaderboard)

| Task | Best public score |
|---|---|
| 1 — Watermark localisation | 0.7872614154749301 |
| 2 — Model equality testing | 0.85 |
| 3 — Unlearning | 0.4700011591814447 |

Sources: the official task statements and opening slides distributed to participants, the event page, and the
organisers' messages.
