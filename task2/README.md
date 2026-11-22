# Task 2 — Model Equality Testing

**Best public score: 0.85** — submission no. 3549, 22/11 at 07:54, file `results/submission_final_0.85.csv`.

## Quick start

```bash
cd task2
pip install -r requirements.txt               # + torch, transformers for the white-box steps 07-08
$EDITOR config.env                            # T2_WORK, T2_MODELS, T2_GPU, PY
SKIP_WHITEBOX=1 bash run_all.sh               # CPU only, ~2 min: rebuilds the submitted CSV
bash run_all.sh                               # everything, white-box scoring on one GPU
```
With `SKIP_WHITEBOX=1`, the pipeline checks three things on the shipped answers, **byte for byte**: the families
(`ep_families2.json`), the black-box split (the 0.50 submission) and the final CSV (0.85).
`COLLECT=1` re-runs the paid collections against the task server instead of using the shipped answers (needs
`API_KEY`; the per-endpoint budget is 100 answers).

## 1. The task

- **33 endpoints** (`ep-01` … `ep-33`), each serving a language model, queried as a black box (prompt in, text out).
- For each one: its **family** (base model) and its **model** (same weights). Not a different model: system prompt,
  persona, watermark, decoding strategy. A different model: fp8 quantisation, LoRA at strength 0.3, 0.7 or 1.5.
- **Score**: `0.1 × MCC_family + 0.9 × MCC_model` over the 528 pairs, each MCC clipped at 0.
- **Constraints**: 100 paid answers per endpoint, 256 tokens max, 4 parallel requests per team, one submission per
  30 min; `temperature` and `seed` ignored, system messages refused.
- Free resource: 9 training endpoints `ref-01` … `ref-09` with known truth (`data/practice_answers.csv`).

## 2. The final partition and the evidence for each endpoint

Family letters: A = LLAMA, B = QWEN_A, C = QWEN_B. A group = one model; any endpoint outside the group is alone.

| Family (checkpoint identified) | Main group | Alone |
|---|---|---|
| **QWEN_A** (Qwen2.5-3B-Instruct) | 01, 02, 04, 12, 23, 33 | 14, 17, 18, 24, 25 |
| **QWEN_B** (Qwen2.5-1.5B-Instruct) | 03, 06, 07, 08, 20, 21, 30 | 22, 26, 28, 32 |
| **LLAMA** (Llama-3.2-1B-Instruct) | 10, 13, 16, 27, 29, 31 | 05, 09, 11, 15, 19 |

`results/endpoint_evidence.csv` gives, for every endpoint, the signal, its value, the script that measures it and the
verdict; `src/decide_partition.py` turns it into the CSV. In short:

- **Cosmetic disguises → main group** (the task says they keep the model):
  - system prompt (recited, or verbose step-by-step preamble): `ep-02`, `ep-08`, `ep-29`;
  - random persona at every request: `ep-23`, `ep-20`, `ep-16`;
  - **watermark** = high rate of identical repeated answers (`src/whitebox/repetition.py`; `ref-06` 0.38 versus
    0.04–0.13 for the other ref endpoints): `ep-04` 0.36, `ep-06` 0.25, `ep-10` 0.20 — the top of each family.
    The black-box test had taken `ep-06` and `ep-10` for different models: the watermark barely changes the token
    distribution, but it makes repeats identical;
  - **temperature 1** fitted in white box (`src/whitebox/prof2.py`, validated on `ref-08`: T = 1.004): `ep-33`,
    `ep-30`, `ep-31`.
- **fp8 → alone**: likelihood ratio against a simulated fp8 model (e4m3, one scale per tensor; `quant.py`, `llr.py`;
  validated on `ref-02`: +19 nats versus −41 to −225 for the other 8): `ep-22` (+2 versus −39 to −79 in QWEN_B),
  `ep-15` (+50 [20; 81] versus −23 to −91 in LLAMA).
- **LoRA → alone**: correlated token-frequency residuals (`resid2.py`; validated: the three ref LoRAs correlate
  0.31–0.75, everything else below 0.2): `ep-05`–`ep-11` (0.29), `ep-17`, `ep-18`, `ep-25` (0.24–0.37); strong
  shifts in the permutation MMD test (`fulltest.py`): `ep-18`, `ep-25`, `ep-28`; LoRA dose-effect axis (`dose.py`):
  `ep-09`.
- **QWEN_A**: black-box split of 22:45 (`solve.py --taille 6`), confirmed by the 01:19 diagnostic (0.55 with only
  QWEN_A split, the value expected from a correct split).
- **Structure from the public diagnostic scores** (`fit.py`, `bayes.py`, scores known before the final submission):
  `ep-03`, `ep-07` in the group (P = 1.00), `ep-13` in (0.62), `ep-26` out (0.31).
- **Judgement calls, no decisive evidence (4 endpoints)**: the scores say exactly one of `ep-21` / `ep-32` is in the
  QWEN_B group (P = 0.50 each) — `ep-21` was chosen; in LLAMA, `ep-27` (P = 0.48) was put in the group and `ep-19`
  (P = 0.57) left alone. They are marked `judgement:*` in the evidence table.

The family part is exact: a deliberate submission with one model label per family scored 0.10 = 0.1 × MCC_family,
so MCC_family = 1.00.

## 3. How we got there

| Time | Public | What changed, and why |
|---|---|---|
| 21/11 21:45 | 0.00 | A single family: no "different" pair, MCC = 0. |
| ~22:15 | 0.05 | Families from the declared publisher, models by distance. A distance on outputs mainly measures **cosmetics**: on the ref endpoints the system prompt (same model) is farther from the plain model (0.730) than fp8 (0.628). No threshold can fix this inverted order. |
| **22:45** | **0.50** | `families2.py`: **3 families of 11** (LLAMA, QWEN_A, QWEN_B) read on habits of the base model (reaction to an injection prefix, knowledge cutoff), not on the declared publisher ("CLAUDE" was a persona). Structure hypothesis 6 cosmetic + 5 weight variants per family: `solve.py --taille 6` picks the most coherent group of 6. |
| 23:15 | 0.10 | Identity probes had entered the distance: they separate the system prompt, not the weights. Manual exclusion list. |
| 23:45 | 0.10 | Deliberate test, one label per family: **MCC_family = 1.00**. Everything left is in the model part. |
| 22/11 01:19 → 02:21 | 0.55; 0.35; 0.10 | **Diagnostics**: one family split as in the 0.50 submission, the other two "all different". QWEN_A right, QWEN_B wrong by one swap, LLAMA worthless. |
| 02:51 → 03:52 | 0.60 (×3) | Long collection (`plan2.py`, 44 requests × 256 tokens per endpoint: fragile questions, equivalent tool choices, arbitrary choices, random sequences). "Indistinguishable by the test" does not mean "same model". |
| **07:54** | **0.85** | **White box on GPU**: checkpoints identified, then watermark, temperature 1, fp8 and LoRA recognised endpoint by endpoint (section 2). |

Rule kept throughout: the leaderboard is a **diagnostic**, not a selector (standard deviation ≈ 0.04–0.07). From 03:52
on, public scores were used to decide between structures the data could not separate; this is within the rules but
is not a standalone method, and the hidden pairs may differ.

### Why the white box unlocked the task

- In black box, fp8 and LoRA 0.3 are undetectable (measured on the ref endpoints; arXiv:2506.06975), and the watermark
  looks like another model. In **teacher forcing** under the right local checkpoint, every answer already paid for
  gives a log-likelihood per token, so precise hypotheses can be tested: temperature, simulated fp8, residuals.
- Checkpoints (`score.py` + `ident.py`, NLL per token): QWEN_B = Qwen2.5-1.5B-Instruct (0.39; 0.5B: 0.82),
  QWEN_A = Qwen2.5-3B-Instruct (0.35; 1.5B: 0.74), LLAMA = Llama-3.2-1B-Instruct with the dated system header (0.63),
  ref = SmolLM2-360M-Instruct (0.94).
- The server samples at pure temperature: no top-p, top-k or repetition-penalty hypothesis fits (`dec.py`).
- Every test was **validated on the ref endpoints** (known truth) before being applied to the ep endpoints.
- The white box costs no request: it only re-reads the answers already collected.

## 4. Pipeline

| Step | Script(s) | Output (in `$T2_WORK`) | Needs |
|---|---|---|---|
| 00 | `steps/00_init_workdir.sh` | copies `data/` and `results/submissions/` | — |
| 01 | `src/official/query_reference.py` | `reference_answers.jsonl` (free ref answers) | `COLLECT=1`, key |
| 02 | `src/blackbox/collect.py` + `probes.py` | `answers_ep.jsonl` (paid) | `COLLECT=1`, key |
| 03 | `src/blackbox/families2.py` | `ep_families2.json` | CPU |
| 04 | `src/blackbox/solve.py --taille 6` | `submission_blackbox.csv` (= the 0.50 submission) | CPU |
| 05 | `src/structure/plan2.py` | `answers_long_ep.jsonl` (paid) | `COLLECT=1`, key |
| 06 | `src/structure/{analyze,fulltest,agree,dose,fit,bayes}.py` | `logs/06_*.log`, posteriors | CPU |
| 07 | `src/whitebox/score.py` (+ `quant.py`) | `whitebox/out/*.pkl` | GPU, checkpoints |
| 08 | `src/whitebox/{ident,dec,prof2,llr,resid2,repetition}.py` | `logs/08_*.log`, `repetition.csv` | CPU |
| 09 | `src/decide_partition.py` | `submission.csv` (= the 0.85 submission) | CPU |
| 10 | `src/structure/submit.py` | server response, archive in `best/` | key |

Steps 03-04 use the **first 966 answers** of `answers_ep.jsonl`: the data available when the 0.50 submission was
built at 22:45. On all 1,672 answers the black-box families change (`ep-23` moves), which is one more reason the
final decision rests on the white-box signals.

## 5. What did not work (measured)

- Averaging 49 probes: MCC 0.000 (the 3 best probes: 0.446). Several questions per request: −44 % discriminative
  power. Strong answer normalisation: less signal.
- Threshold at the largest gap between distances: MCC 0. "Large style gap = cosmetic": false (the largest gaps are
  LoRA 1.5 and 0.7).
- Black-box test "base model up to a temperature": no power. Latency as a side channel: unrelated to the variant.
- KGW z-score with the default `transformers` key: −0.4 on `ref-06`; abandoned.
- Grouping the 8 LLAMA endpoints indistinguishable by the test (03:52): 0.60, false "same" pairs cancel the true ones.
- Isolating the "Claude" endpoints as LoRA (07:22): 0.15, they are personas.

## 6. Files

| Path | Role |
|---|---|
| `config.env`, `run_all.sh`, `steps/`, `slurm/whitebox.sbatch` | configuration and pipeline |
| `src/official/` | official files: task template, reference-endpoint query, practice scoring |
| `src/blackbox/` | probes, paid collection with cache, answer readings, families, split inside families |
| `src/structure/` | long collection (`api.py`, `plan.py`, `plan2.py`), parsing, permutation tests, dose-effect axis, structures compatible with public scores (`fit.py`, `bayes.py`), CSV writing and submission |
| `src/whitebox/` | teacher-forcing scoring, simulated quantisation, checkpoint identification, decoding, temperature, fp8 LLR, residuals, repetition rate |
| `src/decide_partition.py` | final partition from the evidence table |
| `data/` | the answers we paid for (1,672 black-box, 1,453 long), ref answers, probe lists, features |
| `results/` | `submission_final_0.85.csv`, `endpoint_evidence.csv`, `ep_families2.json`, every submitted CSV with its public score (`submissions/`) |

## 7. Notes

- `src/whitebox/repetition.py` and `src/decide_partition.py` were written for this repository: during the hackathon
  the repetition rate was computed inline and the partition was written by hand from the signals. `repetition.py`
  gives 0.36 / 0.25 / 0.20 / 0.38 where the log says 0.39 / 0.23 / 0.17 / 0.37 (the exact probe subset used then was
  not recorded); the ranking inside each family is the same.
- Changes to the hackathon code: data paths read from `$T2_WORK` (`structure/data.py`, `fit.py`, `mkcsv.py`,
  `parse.py`, `plan.py`, `plan2.py`, `archive.py`, `api.py`) and one file name (`answers_ep.jsonl`) in
  `whitebox/score.py`. Inline comments and printed messages are in French (the team's working language).
- The public score covers part of the pairs: 0.85 is not guaranteed on the hidden pairs.

## 8. Generalisation to the hidden pairs

- The final ranking uses the **remaining pairs of the same 33 endpoints**; no new model appears. A placement backed
  by evidence (section 2) is right or wrong on public and hidden pairs alike, so it transfers.
- Identifying the checkpoint of each family is not fitted to labels or scores: it selects among two to four
  candidates by likelihood of the answers, with large gaps (e.g. 0.39 versus 0.82 nat per token for QWEN_B).
- The exposed part is the eight endpoints placed by public scores (`ep-03`, `ep-07`, `ep-13`, `ep-26`) or by judgement
  (`ep-19`, `ep-21`, `ep-27`, `ep-32`). Moving one endpoint changes only a few public pairs and the score is returned
  rounded, so after about twenty submissions the best public score is partly the luckiest: the final score may be
  lower than 0.85 for this reason.
- Two endpoints read differently by our two white-box passes: the CPU pass (`task2-whitebox` branch) saw a
  temperature-1 signature on `ep-28` (alone in the final partition, strong shift in the permutation test) and long
  answers on `ep-29` (in the main group, verbose system-prompt preamble, which also lengthens answers).
