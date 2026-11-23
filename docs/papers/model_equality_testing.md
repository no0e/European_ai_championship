# Model Equality Testing — reading notes (task 2)

**I. Gao, P. Liang, C. Guestrin.** *Model Equality Testing: Which Model Is This API Serving?* ICLR 2025.
[arXiv:2410.20247](https://arxiv.org/abs/2410.20247) · code and dataset:
[i-gao/model-equality-testing](https://github.com/i-gao/model-equality-testing).
The paper is distributed under CC BY 4.0 and is included here: [PDF](gao2025_model_equality_testing.pdf).
Longer course, in French: [`../courses/model_equality_testing_course_fr.pdf`](../courses/model_equality_testing_course_fr.pdf).

## The question

An API claims to serve a given model (say Llama-3 70B). Does it really, or a quantised, watermarked or swapped
version? With only text completions — no weights, no log-probabilities — the paper turns this into a
**two-sample test**: is the distribution of the API's completions the same as the reference model's?

## The test

- Prompts x drawn from a distribution π; completions y of at most L tokens. Samples D_P from the reference, D_Q from
  the API. **H0: π(x)P(y|x) = π(x)Q(y|x).**
- Statistic: the **Maximum Mean Discrepancy** (MMD) with a kernel that factorises per prompt,
  `k((x, y), (x', y')) = 1{x = x'} · k̃(y, y')`.
- p-value by simulation under H0 (sampling the reference again) or by permutation.
- The kernel that wins is the **Hamming kernel**, `k̃(y, y') = Σ_i 1{y_i = y'_i}`: it compares the completions
  **position by position**. With short completions and about ten samples per prompt, almost every full completion is
  unique, so tests on whole strings see no collisions; per-position comparisons do, especially on the first tokens,
  where the distributions are concentrated and where quantisation or a watermark moves mass.

## Key findings

- With about 10 samples per prompt, median power of **77.4 %** against quantisation and watermarking, at α = 0.05.
- **fp16 is essentially undetectable** (power near zero), so the authors put fp16 in a composite null hypothesis.
- Changes of temperature are detected almost always.
- **Audit**: 31 endpoints serving Llama models across 9 providers (summer 2024); **11 of 31** were rejected as not
  matching the reference weights, for less than $1 per audit.
- The MMD also works as a **distance** between models, which groups models by family.
- Limits: the test needs a **reference model to sample from**; it says "different", not *why*; a one-token shift
  misaligns every later position of the Hamming kernel.

## What we used for task 2, and where we departed from it

- **The framing**: each pair of endpoints is a hypothesis test "same weights?". We kept the permutation MMD test as
  one of our tools (`task2/src/structure/fulltest.py`): it sees strong weight variants.
- **The trap of the task**: the scored deployments differ by system prompts, personas, watermarks and decoding
  strategies, which change the text far more than a weight change does. Measured on the practice endpoints, a system
  prompt moves an endpoint further from its base (distance 0.730) than an fp8 quantisation (0.628). A black-box
  distance alone groups the wrong endpoints together.
- **The step the paper assumes**: MET supposes access to the reference model. We **identified the checkpoint** of each
  family by likelihood — re-scoring the answers we had already paid for, token by token, on 2 to 4 local candidates
  (Qwen2.5-1.5B, Qwen2.5-3B, Llama-3.2-1B) — then used it as a common reference for all endpoints of the family.
- **One test per kind of variant**, each validated on the `ref` endpoints: a likelihood ratio against our own simulated
  fp8 (the most powerful test when both hypotheses are known, by Neyman–Pearson), a fitted temperature, correlated
  token-frequency residuals for LoRA, the rate of identical answers for watermarks.
- Details: [`task2/README.md`](../../task2/README.md); method explained in French:
  [`../courses/task2_method_explained_fr.md`](../courses/task2_method_explained_fr.md).

## Other references given with the task

- X. Zhu et al., *Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test* (RUT), ICLR 2026 —
  [arXiv:2506.06975](https://arxiv.org/abs/2506.06975).
- X. Wang et al., *AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes*, ICLR 2026 —
  [arXiv:2609.00052](https://arxiv.org/abs/2609.00052).
