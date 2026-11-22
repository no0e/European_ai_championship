# Sevel Labs ENSAI — European AI Championship Paris 2026

Team **Sevel Labs ENSAI**. Code of our best submission for each of the three tasks of the Paris stage
(21–22 November 2026). Each task folder is self-contained: one configuration file to edit, one script that runs the
whole pipeline from the official data to the submitted file, and a README with the method, how we got there, what did
not work, and reproducibility notes.

- **Our solutions, in detail**: [task 1](task1/README.md) · [task 2](task2/README.md) · [task 3](task3/README.md)
- **Pointers to the code that solves each task**: [Where to look](#where-to-look-task-by-task), below
- **Slides of the presentation**: [`slides/Slides Sevel Labs.pdf`](slides/Slides%20Sevel%20Labs.pdf)
- **The two model files we submitted**: release [`submitted-files`](../../releases/tag/submitted-files)

| Task | Best public score | Submitted file | Method in one line |
|---|---|---|---|
| [Task 1 — Watermark localisation](task1/README.md) | **0.7872614154749301** | `sub_m5.pt` | one dense detector per method, trained on exact composites made with the official embedders (on a **watermarked frame larger than the image**), second scale, CV fine-tuning on real images, ensemble + guided filter + TrustMark rectangles |
| [Task 2 — Model equality testing](task2/README.md) | **0.85** | `results/submission_final_0.85.csv` | families read on habits of the base model, then **white box**: local checkpoints identified, watermark / temperature / fp8 / LoRA recognised by teacher forcing of the answers already collected |
| [Task 3 — Unlearning](task3/README.md) | **0.4700011591814447** | `plat6.pt` | rank-based margin targets + distillation matched on per-image loss, **weight average** of six models |

## Where to look, task by task

Each row is a step that matters and the exact place it lives. Paths are relative to the task folder.

### Task 1 — watermark localisation (`task1/`)

| What it does | Where |
|---|---|
| Official metric: per-image mIoU, 512 → 128 downsampling, stratified folds, top-k decision | `src/t1_metric.py` — `miou_score`, `downsample_mask`, `stratified_folds`, `topk_decision` |
| Regime (whole image watermarked, or a small area) from the message log-likelihood | `src/compute_signals.py` writes the signals; `src/dense/common.py` — `global_ll`, `best_split`, `regime_thresholds` reads them |
| **The founding idea**: exact synthetic pairs built with the official embedders, watermarked on a frame *larger* than the image | `src/dense/gen_pairs.py` — `base_pool`, `add_rects`, options `--frame`, `--frame_max`, `--with_full` |
| The dense detector: the watermark decoder as backbone, FPN, full-resolution branch, 128×128 output | `src/dense/common.py` — `DenseDet`, and `MaskSampler` for the realistic hole statistics |
| Training, CV fine-tuning, pseudo-labels of the test, second scale | `src/dense/train_dense.py` — `SynthSet`, `RealSet`, `pseudo_labels`, `load_frames`, flag `--scale2` |
| Ensemble, guided filter and decision rule chosen **out-of-fold** per (method, regime) | `src/dense/assemble.py` — `load_run`, `post_map`, `guided_color`, `apply_rule`, `Regime`, `score_table` |
| TrustMark rectangles, found by re-reading the message inside a crop | `src/rect_search.py` — `grid_boxes`, `coarse_scores`, `refine`, `plateau_center` |
| Paired CV comparison, which decided every change | `src/dense/compare_cv.py` |
| Run it | `run_all.sh` (steps 00–12) → `out/sub_m5.pt`; the ensemble itself is `steps/12_assemble.sh` |

### Task 2 — model equality testing (`task2/`)

| What it does | Where |
|---|---|
| Families, read on habits of the base model rather than on what it claims to be | `src/blackbox/families2.py` — `profils`, `familles`; the probes are `src/blackbox/probes.py` |
| Black-box split inside a family (most coherent group of six) | `src/blackbox/solve.py` — `persona`, `stabilite` |
| **The jump to 0.85**: re-scoring answers already paid for, token by token, on local checkpoints | `src/whitebox/score.py` — `load_rows`, `build_prompt`; checkpoint identification in `src/whitebox/ident.py` |
| fp8 quantisation, caught by a likelihood ratio against our own simulated fp8 | `src/whitebox/quant.py` — `f8`, `apply`; the ratio itself in `src/whitebox/llr.py` |
| LoRA adapters, caught from correlated token-frequency residuals | `src/whitebox/resid2.py` — `tmle` |
| Temperature fitted per endpoint | `src/whitebox/prof2.py` — `tmle` |
| Watermarks, caught from repeated identical answers | `src/whitebox/repetition.py` — `rate` |
| Structures compatible with the public scores, and the posterior of each endpoint | `src/structure/bayes.py` — `space`, `pred_vec`, `score_arr`; the tests themselves are `src/structure/analyze.py`, `src/structure/fulltest.py`, `src/structure/agree.py`, `src/structure/dose.py`, `src/structure/fit.py` |
| The final partition, and its byte-for-byte check against what we submitted | `src/decide_partition.py` — `blackbox_groups`, `main` |
| Run it | `SKIP_WHITEBOX=1 bash run_all.sh` — two minutes on CPU, rebuilds `results/submission_final_0.85.csv` exactly |

### Task 3 — unlearning (`task3/`)

| What it does | Where |
|---|---|
| **The founding idea**: one target per image, the true-class logit lowered by a margin set by the image's difficulty rank | `src/tg.py` — `margin`, `rank01`, `difficulty`, `build` (the formula `m(x) = mu − sigma·Φ⁻¹(u(x)) − beta·rho_c`) |
| Distillation towards those targets, with a Huber loss on the per-image cross-entropy | `src/k3.py` — `run_distill`, `kl_t`, `cap_targets`, `displaced` |
| Altered copies of the forget images (they carry half the score), pushed further | `src/k3.py` — `aug_real`, `aug_strong`, `aug_broad`, `rand_views`, option `--view_delta` |
| Anchors that hold the model in place everywhere else | `src/prepare_anchors.py` builds them; `src/k3.py` — `anchor_batch` draws each batch |
| The simulator that told us what a retrained model looks like | `src/k3.py` — `Env`, `SimEnv`, `RealEnv`, `get_env` |
| Weight averaging of the six models of one plateau | `src/k3.py` — `run_avg` (`python src/k3.py avg`) |
| Run it | `run_all.sh` (steps 01–04) → `models/plat6.pt` |

## Layout of every task folder

```
taskN/
  README.md        method, how we got there, pipeline, what did not work, reproducibility notes
  config.env       paths (data, models, outputs), GPUs, Python interpreter -- the only file to edit
  run_all.sh       runs every step in order
  steps/           one short script per step, numbered (each can be re-run alone)
  src/             Python code
  slurm/           example batch script for a Slurm cluster
  results/         what we submitted and the scores obtained
  requirements.txt
```
Task 2 also ships `data/`: the answers we paid for (they cannot be bought again), so its decision steps run on CPU in
about two minutes and rebuild the submitted CSV byte for byte.

## Running

```bash
cd taskN && pip install -r requirements.txt && $EDITOR config.env && bash run_all.sh
```

### Checking our scores without retraining

The two model files we submitted are attached to this repository's GitHub release `submitted-files`, so the public
scores can be obtained again by submitting them, with no GPU and no training:

```bash
gh release download submitted-files -D /tmp/subs        # sub_m5.pt, plat6.pt
# no release on this copy of the repository? use your own copies of the two files below,
# or rebuild them with run_all.sh in task1/ and task3/.

cd task1 && WMLOC_API_KEY=... bash steps/13_submit.sh /tmp/subs/sub_m5.pt "m5 re-check"
cd task2 && API_KEY=...       bash steps/10_submit.sh "$PWD/results/submission_final_0.85.csv"
cd task3 && T3_API_KEY=...    bash steps/05_submit.sh /tmp/subs/plat6.pt
```

Expected: **0.7872614154749301**, **0.85**, **0.4700011591814447**.

Two traps worth knowing. **Give absolute paths**: steps 13 (task 1) and 10 (task 2) change into the working
directory before handing the file to Python, so a relative path is resolved against `$T1_WORK` / `$T2_WORK` and not
found. And each server rate-limits: task 2 accepts one submission per 30 minutes, task 1 answers HTTP 429 when it is
too early — the task 1 script keeps its own journal in `$T1_WORK/out/submissions.csv` either way. Task 3 reads its
key from `$T3_API_KEY`; without it, `submit.py` falls back to a file that does not exist here and raises
`FileNotFoundError`.

| Task | Data to provide | Hardware we used | Time (estimate) |
|---|---|---|---|
| 1 | official `train.zip` / `test.zip`, DIV2K validation HR, ConvNeXt-tiny and SAM ViT-H weights | 1 node, 4 × A100 40 GB | ~6–8 h |
| 2 | nothing for steps 00–06 and 09 (answers shipped); Hugging Face checkpoints for the white-box steps | CPU; 1 GPU for steps 07–08 | 2 min CPU; ~1 h GPU |
| 3 | official `model.pt` and `forget_images.npz`, COCO val2017 images | 1 node, 4 × A100 | ~20 min |

Submission scripts read the team key from the environment (`WMLOC_API_KEY`, `API_KEY`, `T3_API_KEY`); no key is
stored in this repository.

## Notes

- The two submitted model files are the ones the servers scored: `sub_m5.pt` (task 1, 16,734,551 B, sha1
  `317a0ead1eee7df22bb1867ae68526c0a04d2ff7`) and `plat6.pt` (task 3, 343,617,075 B, sha1
  `978e7668096ee8affd463ac1b6b2ac58da87ff8e`). Both were copied off the JURECA scratch after the event. `plat6.pt`
  is far above the Git file size limit, so both are attached to this repository's release
  [`submitted-files`](../../releases/tag/submitted-files); `run_all.sh` also rebuilds them in `task1/` and `task3/`,
  close but not identical. The sha1 above identify the exact files that were scored. The task 2 submission is a CSV
  and is in the repository, at `task2/results/`.
- Scores are the public leaderboard values (part of the test data); the final ranking uses the hidden part.
- Training in tasks 1 and 3 is stochastic: a re-run gives close but not identical scores. The released files are the
  ones to use to check our scores directly, without retraining; each `results/` folder states what every submission
  contained. Task 2 needs no file — its decision steps rebuild the submitted CSV byte for byte.
- **What is not included, and does not block anything.** Three options of the shipped code need a module we did not
  ship, and no step of any pipeline passes them: `rect_search.py --ss_torch` (wants `onnx2torch`; the submission only
  searches TrustMark rectangles), `score.py --lora` (wants a scratch `lora_util.py`; the LoRA detection we submitted
  reads token-frequency residuals instead, `--watch 3000` then `resid2.py`), and `k3.py --env sim` (wants a `sim/`
  folder holding the CIFAR-100 simulator; everything submitted runs `--env real`). Everything `run_all.sh` touches is
  here.
- The code is the hackathon code, reorganised: paths became configurable and the cluster-specific launchers were
  replaced by the numbered steps. Every change is listed at the end of each task README. Inline comments and printed
  messages are mostly in French, the team's working language.
