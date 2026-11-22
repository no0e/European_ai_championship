# Task 3 — Unlearning (make a ViT forget 3,500 images)

**Best public score: 0.4700011591814447** — submitted 22/11 at 07:59, file `plat6.pt`: the parameter-wise
**average of six unlearned models** (each one alone scored 0.4591–0.4646).

## Quick start

```bash
cd task3
pip install -r requirements.txt            # torch, timm, numpy, requests (+ diffusers, transformers for step 02)
$EDITOR config.env                         # T3_WORK (must contain model.pt and forget_images.npz), COCO_DIR, GPUS
bash run_all.sh                            # steps 01-04 -> $T3_WORK/models/plat6.pt
bash steps/05_submit.sh                    # optional: submit (needs T3_API_KEY)
```
Hardware used: one node with 4 × A100 (JURECA-DC). Each of the six models takes about 6 minutes on one GPU;
step 02 takes about 2 minutes on 4 GPUs. `slurm/run_all.sbatch` is an example batch script.

## 1. The task

- **Model**: ViT-B/16 classifying 100 classes (`model.pt`, "orig"). **Forget set**: 3,500 images
  (`forget_images.npz`), **without labels**. No retain set and no held-out set are given.
- **Submission**: the `state_dict` of the unlearned model. The server compares it with "gold" (models retrained
  without the forget set):
  - `forget = mean_std_score × prediction_score`: mean and standard deviation of the loss on the forget images must
    match gold's, and so must the per-image outputs at T = 2 (KL(gold‖sub) / KL(gold‖orig));
  - `retain = accuracy_score × kl_similarity` on a hidden control set: stay close to orig elsewhere;
  - score = forget × retain, and **0 if the hidden accuracy falls below 70 %**.
- Submitting orig unchanged scores 0. The goal is not to forget as much as possible but to **imitate a model that
  never saw these images**.

## 2. Method

**Idea in three sentences.** On the forget images, gold behaves like a model at test time: often wrong, with large
and widely spread losses. So for every image we build a **target** = orig's logits in which only the logit of the
original class is lowered, the more so the "harder" the image. Then we **distil**: the student starts from orig,
learns these targets on the forget images and on altered copies of them, and stays equal to orig on anchor images.

### 2.1 Rank-based margin targets — `src/tg.py`

- Original class y = orig's prediction. Difficulty u(x) ∈ [0, 1] = rank of orig's mean margin over the clean image
  and 4 deterministic augmented views (`--rank mv`).
- Imposed margin of the original class: `m(x) = mu − sigma · Φ⁻¹(u(x)) − beta · rho_c`, where rho_c = n_c / max n_c is
  the relative size of the class in the forget set. Logits of the other classes = orig's, scaled by `tb`.
- Selected setting: `mu = −2.5`, `sigma = 7`, `tb = 1.4` (push the other classes slightly apart),
  `--pos_scale 0.5` (positive margins halved: images that gold still classifies correctly are less confident).

### 2.2 Distillation — `src/k3.py distill`

- 30 epochs, lr 1e-4, start = orig; KL at T = 2 towards the targets **plus a Huber loss on the per-image
  cross-entropy** (`--w_ce 1`; without it the tail of the loss distribution is not reproduced).
- Altered copies of the forget images (crop, flip, colour, blur, noise) — they count for half of the score; their
  margin is lowered by 4.5 more (`--view_delta 4.5`), with 30 % clean images (`--clean_frac`) and 20 % strong views
  (`--strong_frac`).
- Anchors: 5,000 COCO val2017 images, on which the student must keep orig's outputs (KL).

### 2.3 Weight averaging — `src/k3.py avg`

Six models from the same plateau of settings, all starting from orig, averaged parameter by parameter. Averaging
removes the training noise of each run (±0.006, measured by re-running one setting with another seed) and gives the
real level of the setting: **0.4700** versus 0.459–0.465 for each model. Averaging twelve models gave 0.4690.

| Model | What distinguishes it | Public score alone |
|---|---|---|
| `ps1` | reference setting (`--pos_scale 0.5`) | 0.4605 |
| `ps10` | `ps1` with another seed | not submitted |
| `st1` | 35 % strong views, 25 % clean images | 0.4646 |
| `bt1` | class-size dependent margin, `beta = 2`, `mu = −1.1` | 0.4591 |
| `bt2` | same, `beta = −2`, `mu = −3.9` | 0.4597 |
| `y1` | 30 % synthetic variants (diffusion) treated as copies to forget | 0.4596 |

## 3. How we got there

| Time | Best public | What changed, and why |
|---|---|---|
| 21/11 17:13 | 0.024 | Uniform output softening (T = 8) + COCO anchors: gold is not "less confident everywhere". |
| 21/11 afternoon | 0 (×2) | Head surgery on the 9 "heavy" classes: harmful. |
| 21/11 22:58 | 0.1496 | SCRUB folds extrapolated in weight space: forgets little, drifts a lot (anchor agreement 0.61). |
| 22:00–22:30 | — | Restart from scratch. The model was trained from scratch (cosine ≈ 0 with 7 public timm checkpoints); images are diffusion-generated, fine classes grouped in families. |
| 22:40–23:30 | — | **Simulator**: ViT trained from scratch on CIFAR-100, same forget-set structure, 4 golds, exact scoring. It gave the **structure** of the method: set the original-class margin by difficulty rank (0.37–0.46 forget score versus 0.04 for softening), and add the per-image CE loss. |
| **23:37** | **0.2585** | First k3 submission. |
| 22/11 00:48 | 0.3524 | Margin calibration: optimum near mu −3, sigma 7. |
| 02:33 → 03:02 | 0.381 | The student forgot **less** on altered copies (loss 3.18 on a crop, 4.01 on a flip, versus 4.58 on the clean image) whereas gold loses rather more there. Fix: `--view_delta` + 70 % views. |
| 03:21 → 03:32 | 0.4484 | Stop squeezing the other classes' logits (`tb 1.0`: 0.4307), then push them apart (`tb 1.4`). The simulator said the opposite: its fine settings do not transfer. |
| 03:51 → 05:56 | — | Axis-by-axis check: local optimum (mu −1.5: 0.4189; −3.5: 0.3745; 50 epochs: 0.4027; 20: 0.4289). Noise measured at ±0.006. |
| 06:06 | 0.4605 | `--pos_scale 0.5` (`ps1`). 0.3: 0.4557; 0: 0.4365. |
| 07:19 | 0.4646 | 35 % strong views (`st1`), within noise. |
| **07:59** | **0.4700** | **Weight average of six plateau models (`plat6`).** |

What we learned about the problem:
1. Gold is "at test time" on the forget images: a simple model of the score, fitted on seven submissions, puts its
   mean loss around 5 ± 4.5.
2. Gold knows every class, including the 9 classes with 130 forget images each (`h1` 0.1382 versus 0.3455).
3. The 70 % barrier is far: disabling 3 classes costs only 5 % of the score.
4. Altered copies count for half, and a naive student forgets less on them than on the clean image.
5. Touching only the original-class logit is enough: squeezing the other logits hurts, spreading them helps.
6. Retention is a structural cost: on the simulator, neither in-domain anchors, adversarial negatives nor parameter
   subsets raise it by more than 5 %.

## 4. Pipeline

| Step | Script | Output (in `$T3_WORK`) |
|---|---|---|
| 01 | `steps/01_prepare_anchors.sh` → `src/prepare_anchors.py` | `anchors.npz` (5,000 COCO images, 224 px) |
| 02 | `steps/02_synthetic_variants.sh` → `src/gen_variants.py`, `src/merge_variants.py` | `k5/syn_s0.5.npz` (2 variants per forget image; used by `y1` only) |
| 03 | `steps/03_train_models.sh` → `src/tg.py`, `src/k3.py distill` | `targets/<name>.pt`, `models/<name>_a1.0.pt`, `logs/` |
| 04 | `steps/04_weight_average.sh` → `src/k3.py avg` | `models/plat6.pt` (submitted file) |
| 05 | `steps/05_submit.sh` → `src/submit.py` | server response, `output/submit_log.txt` |

The options in step 03 are those of the hackathon runs, read from the logs of the job queue that produced the six
models; those raw logs are not part of this repository. Each distillation prints its own local statistics (accuracy
and loss on the forget images, agreement and KL on the anchors), which is what we compared during the night.

## 5. What did not work (measured)

| Idea | Verdict | Measurement |
|---|---|---|
| Uniform output softening | insufficient | 0.024–0.027 public |
| Head surgery on the 9 heavy classes | harmful | 0.0 (two tries) |
| SCRUB folds as targets, extrapolated | superseded | 0.15; strong drift |
| Neighbour-purity difficulty ranking (mid-layer features) | does not transfer | better on the simulator, −9 % on the real task |
| Gold never saw the 9 heavy classes | refuted | `h1` 0.1382 vs 0.3455 |
| Squeezing the other logits (tb 0.8) | harmful | 0.381 vs 0.4307 (tb 1.0) |
| 50 or 20 epochs instead of 30 | worse | 0.4027 and 0.4289 vs 0.4484 |
| Zero margin for correctly classified images (`--pos_scale 0`) | worse | 0.4365 vs 0.4605 |
| Strong views kept equal to orig | worse | 0.4268 |
| Same wrong class for all copies of an image | harmful | 0.3445 |
| Fisher-weighted penalty (certified removal) | no effect | simulator 0.361 vs 0.360 |
| Synthetic variants as in-domain anchors | harmful | 0.3995 |
| 50 % strong views | slightly worse | 0.4508 |

Lesson: any protection placed close to the forget images (strong views kept, near variants as anchors) prevents
forgetting from spreading to their transformed versions, which count for half of the score.

## 6. Files

| Path | Role |
|---|---|
| `config.env` | paths, GPUs, submission URL — the only file to edit |
| `run_all.sh`, `steps/` | the pipeline, one script per step |
| `slurm/run_all.sbatch` | example batch script for a 4-GPU node |
| `src/task_template.py` | official template: ViT builder, server preprocessing, data loading |
| `src/prepare_anchors.py` | COCO anchors |
| `src/gen_variants.py`, `src/merge_variants.py` | synthetic variants (sd-turbo) and shard merging |
| `src/tg.py` | rank-based margin targets |
| `src/k3.py` | real environment, augmented views, distillation (`distill`), weight averaging (`avg`); also contains the simulator and out-of-fold modes, not used here |
| `src/submit.py` | submission with server rate-limit handling |
| `results/` | scores of every model in `plat6` and the final submission |

## 7. Notes on reproducibility

- Training uses random augmentations and bf16: a re-run lands within about ±0.006 of each single-model score; the
  average is more stable.
- Two steps were reconstructed for this repository: merging the variant shards (`src/merge_variants.py`, a one-off
  command during the hackathon) and the targets of `y1` (no `tg.py` log exists for it; the step copies the targets of
  `ps1`, which has the same target options).
- Changes to the hackathon code: working directory read from `$T3_WORK` (`k3.py`, `gen_variants.py`, `submit.py`),
  key and URL read from the environment (`submit.py`). Options added to `k3.py` after `ps1` (`--ewc`, `--syn_frac`,
  `--llrd`, `--ema`, `--others_src`) are inactive by default. Inline comments in the code are in French (the team's
  working language).

## 8. Limitations and generalisation to the final ranking

**What the targets assume.** The margin rule `m(x) = mu − sigma · Φ⁻¹(u(x))` is a quantile transform: images are
ranked by difficulty and given the quantiles of a normal law N(mu, sigma²), so the imposed margins follow that law in
the order of difficulty. With the selected `mu = −2.5`, `sigma = 7`, the share of forget images whose imposed
margin stays positive is P(m > 0) = Φ(mu / sigma) = Φ(−0.36) ≈ **0.36** — the targets
implicitly treat gold as a model with a test accuracy of roughly a third on these images. This number was never
measured; it follows from settings tuned on the public score.

**What comes from reasoning and what comes from the leaderboard.**

| Part | Origin |
|---|---|
| "Forgetting = behaving at test time", rank-based margins, per-image CE term, lowered margin on altered copies, COCO anchors, weight averaging | simulator (known gold), a local diagnostic (`view_stats`), the scoring formula, measured training noise |
| Values of `mu`, `sigma`, `tb`, `view_delta`, `pos_scale`, number of epochs, share of strong views | axis-by-axis search on the public score (92 team submissions in total) |
| "Gold's loss on the forget images is about 5.2 ± 4.5" | a simple model of the score fitted on seven public scores |

The simulator gave the structure but not the fine settings: it pointed the wrong way for `tb` and for the
neighbour-purity ranking.

**Why the leaderboard tuning should transfer, and its limit.**
- The tuned quantities are a handful of **global** settings applied to every image, not per-image decisions: few
  degrees of freedom to fit the noise of the public part.
- Public (about 30 %) and final (about 70 %) scoring data are split by a hash of each image identifier, so both are
  drawn from the same population.
- Limit: the best public scores (0.4605, 0.4646, 0.4700) differ by about the training noise (±0.006), and the
  sampling noise of a 30 % evaluation subset was not measured. Choosing the best of many submissions favours the
  luckiest one, so the final score may be somewhat lower. The weight average is the least exposed choice: it does not
  depend on one lucky run.
