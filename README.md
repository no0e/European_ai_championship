# European AI Championship Paris 2026 — our solutions

Code of our best submission for each of the three tasks of the Paris stage (21–22 November 2026). Each task folder is
self-contained: one configuration file to edit, one script that runs the whole pipeline from the official data to the
submitted file, and a README with the method, how we got there, what did not work, and reproducibility notes.

| Task | Best public score | Submitted file | Method in one line |
|---|---|---|---|
| [Task 1 — Watermark localisation](task1/README.md) | **0.7872614154749301** | `sub_m5.pt` | one dense detector per method, trained on exact composites made with the official embedders (on a **watermarked frame larger than the image**), second scale, CV fine-tuning on real images, ensemble + guided filter + TrustMark rectangles |
| [Task 2 — Model equality testing](task2/README.md) | **0.85** | `results/submission_final_0.85.csv` | families read on habits of the base model, then **white box**: local checkpoints identified, watermark / temperature / fp8 / LoRA recognised by teacher forcing of the answers already collected |
| [Task 3 — Unlearning](task3/README.md) | **0.4700011591814447** | `plat6.pt` | rank-based margin targets + distillation matched on per-image loss, **weight average** of six models |

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

- The two submitted model files are attached to this repository's GitHub release `submitted-files`, with their
  sha1: `sub_m5.pt` (task 1,
  16,734,551 B, sha1 `317a0ead1eee7df22bb1867ae68526c0a04d2ff7`) and `plat6.pt` (task 3, 343,617,075 B, sha1
  `978e7668096ee8affd463ac1b6b2ac58da87ff8e`, far above the Git file size limit). They are the files the servers
  scored; both were copied off the JURECA scratch after the event. The task 2 submission is a CSV and is in
  `task2/results/`.
- Scores are the public leaderboard values (part of the test data); the final ranking uses the hidden part.
- Training in tasks 1 and 3 is stochastic: a re-run gives close but not identical scores. The released files are the
  ones to use to check our scores directly, without retraining; each `results/` folder states what every submission
  contained. Task 2 needs no file — its decision steps rebuild the submitted CSV byte for byte.
- The code is the hackathon code, reorganised: paths became configurable and the cluster-specific launchers were
  replaced by the numbered steps. Every change is listed at the end of each task README. Inline comments and printed
  messages are mostly in French, the team's working language.
