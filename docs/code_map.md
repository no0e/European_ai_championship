# Code map — where each idea lives

Each row is a step that matters and the exact place it lives. Paths are relative to the task folder.

## Task 1 — watermark localisation (`task1/`)

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

## Task 2 — model equality testing (`task2/`)

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

## Task 3 — unlearning (`task3/`)

| What it does | Where |
|---|---|
| **The founding idea**: one target per image, the true-class logit lowered by a margin set by the image's difficulty rank | `src/tg.py` — `margin`, `rank01`, `difficulty`, `build` (the formula `m(x) = mu − sigma·Φ⁻¹(u(x)) − beta·rho_c`) |
| Distillation towards those targets, with a Huber loss on the per-image cross-entropy | `src/k3.py` — `run_distill`, `kl_t`, `cap_targets`, `displaced` |
| Altered copies of the forget images (they carry half the score), pushed further | `src/k3.py` — `aug_real`, `aug_strong`, `aug_broad`, `rand_views`, option `--view_delta` |
| Anchors that hold the model in place everywhere else | `src/prepare_anchors.py` builds them; `src/k3.py` — `anchor_batch` draws each batch |
| The simulator that told us what a retrained model looks like | `src/k3.py` — `Env`, `SimEnv`, `RealEnv`, `get_env` |
| Weight averaging of the six models of one plateau | `src/k3.py` — `run_avg` (`python src/k3.py avg`) |
| Run it | `run_all.sh` (steps 01–04) → `models/plat6.pt` |
