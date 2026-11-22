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

| Task | Data to provide | Hardware we used | Time (estimate) |
|---|---|---|---|
| 1 | official `train.zip` / `test.zip`, DIV2K validation HR, ConvNeXt-tiny and SAM ViT-H weights | 1 node, 4 × A100 40 GB | ~6–8 h |
| 2 | nothing for steps 00–06 and 09 (answers shipped); Hugging Face checkpoints for the white-box steps | CPU; 1 GPU for steps 07–08 | 2 min CPU; ~1 h GPU |
| 3 | official `model.pt` and `forget_images.npz`, COCO val2017 images | 1 node, 4 × A100 | ~20 min |

Submission scripts read the team key from the environment (`WMLOC_API_KEY`, `API_KEY`, `T3_API_KEY`); no key is
stored in this repository.

## Notes

- Scores are the public leaderboard values (part of the test data); the final ranking uses the hidden part.
- Training in tasks 1 and 3 is stochastic: a re-run gives close but not identical scores. The submitted files are
  provided to check our scores directly (see each `results/` folder).
- The code is the hackathon code, reorganised: paths became configurable and the cluster-specific launchers were
  replaced by the numbered steps. Every change is listed at the end of each task README. Inline comments and printed
  messages are mostly in French, the team's working language.
