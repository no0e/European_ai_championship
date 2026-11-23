# Task 2 — Model equality (official statement, summarised)

Supervisors: Adam Dziedzic and Franziska Boenisch. Task developed by Xun Wang, Bihe Zhao, Maitri Shah, Ivo Hoese and
Aditya Kumar (CISPA / SprintML). Our solution: [`task2/README.md`](../../task2/README.md).

## Background

Model equality testing asks whether two APIs serve the same model, by comparing their answers to well-chosen inputs.
Since one model can answer the same prompt differently, a useful test must separate real differences in behaviour
from ordinary sampling noise.

## Setting

- **33 deployments** of language models, each behind its own endpoint, `ep-01` … `ep-33`.
- A deployment may differ in its **decoding watermark**, a **provider-added system prompt**, its **decoding
  strategy**, its **quantisation** or a **LoRA adapter**.
- Two endpoints serve **the same model** when they use the same checkpoint, the same quantisation and the same LoRA
  adapter. Watermark, system prompt and decoding strategy do **not** change model identity; base model, quantisation
  and LoRA adapter **do**. Sharing a family alone does not make two models equal.

The challenge: system prompts, decoding strategies and watermarks can change the text of one model a lot, while a
quantised or LoRA variant can answer almost like its base model.

**Goal**: for every endpoint, its **family** (endpoints built on the same base model); for every pair, whether they
serve **the same model**.

## Metric

Matthews correlation coefficient over the **528 unordered pairs**, clipped at 0:

    MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    score = 0.1 × MCC_family + 0.9 × MCC_model

`MCC_family` is computed over all pairs ("same family?"); `MCC_model` over the pairs inside one family only ("same
model?"). Finding the families alone is therefore worth at most 0.1: the score is in telling the variants apart
inside each family.

## API and budget

- **100 answers per endpoint for the whole competition**; a request for several answers costs one per answer.
- At most 256 generated tokens per answer and 1,024 input tokens per prompt; at most 4 parallel requests per team.
- `temperature` and `seed` are ignored; system messages are rejected; invalid requests are not counted.
- **Practice endpoints** `ref-01` … `ref-09`: a separate small model with the same kinds of disguises and traps, with
  the answer published (`practice_answers.csv`) and a local scorer (`score_practice.py`). The official helper
  scripts are shipped in [`task2/src/official/`](../../task2/src/official/).

## Submission

A CSV with one line per endpoint, `endpoint,family,model`; free-text labels; endpoints sharing a `model` label must
also share the `family` label. **One submission every 30 minutes** (failed ones included).

The statement insists, in capitals: **do not rely on the public score or optimise only for it.** The leaderboard
scores only part of the pairs; the final ranking uses the remaining, unseen pairs, and "a submission that is tuned to
the leaderboard can score high there and close to nothing in the final ranking".

## References given with the task

- I. Gao, P. Liang, C. Guestrin, *Model Equality Testing: Which Model Is This API Serving?*, ICLR 2025 —
  [arXiv:2410.20247](https://arxiv.org/abs/2410.20247). Our notes: [papers/model_equality_testing.md](../papers/model_equality_testing.md);
  the paper itself: [PDF](../papers/gao2025_model_equality_testing.pdf).
- X. Zhu et al., *Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test*, ICLR 2026 —
  [arXiv:2506.06975](https://arxiv.org/abs/2506.06975).
- X. Wang, B. Zhao, M. Backes, F. Boenisch, A. Dziedzic, *AgentProv: Auditing Agentic LLM API Providers via
  Tool-use Policy Probes*, ICLR 2026 — [arXiv:2609.00052](https://arxiv.org/abs/2609.00052).

## What the statement implied for us

- **Cosmetic changes dominate the text**, weight changes are subtle: a naive black-box distance groups the
  disguised endpoints together and the base model with its weight variants — the opposite of what is scored.
- The budget is fixed (3,300 answers in total), so every answer had to be kept: they are shipped in
  [`task2/data/`](../../task2/data/), and the white-box analysis re-reads them on local checkpoints at no query
  cost.
- The `ref` endpoints are the only place where a test can be validated against a known truth; every test we used
  was checked there first ([`task2/README.md`](../../task2/README.md) § 2).
