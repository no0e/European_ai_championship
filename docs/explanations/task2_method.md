# Task 2 — our method, explained and criticised

> Written during the event (22/11, around 10:45) from the team's working logs, to understand what we had done,
> separate what is a method from what came from the leaderboard, and be able to explain it. Figures are those of the
> logs; nothing was re-run for this note. The task itself: [statement](../tasks/task2_model_equality.md) ·
> [solution README](../../task2/README.md).

## 1. The problem in one sentence

33 API endpoints serve language models. Say which ones are **the same model** (same weights), ignoring **cosmetic**
disguises (system prompt, persona, watermark, temperature) and spotting **weight changes** (fp8 quantisation, LoRA
adapters at several strengths). Score = 0.1 × MCC_family + 0.9 × MCC_model, over **pairs** of endpoints. Budget: 100
answers per endpoint for the whole competition.

It is a cascade of hypothesis tests: for every pair, H₀ = "same weights".

## 2. The method, step by step

### Step 1 — Families (MCC_family = 1.00)

Endpoints say who they are ("I'm Llama", "I'm Qwen") and pass or fail the same simple tasks. Three families of 11:
LLAMA, QWEN_A, QWEN_B. Easy, and worth only 10 % of the score.

### Step 2 — The trap: cosmetics dominate

On the practice endpoints (`ref`, known truth), a system prompt moves an endpoint further from its base (distance
0.730) than fp8 quantisation (0.628). **Consequence**: a naive black-box distance groups the disguised endpoints
together, and the base model with its weight variants — exactly the opposite of what is scored. First attempt: 0.05.

**An example to make it concrete**: the same model told "you are a pirate" answers very differently from itself
without instructions, while its fp8 version answers almost like it. The distance measures the *form* of the
answers, not the weights.

### Step 3 — Identify the exact model of each family (white box)

For every answer already paid for, we compute its probability under downloaded candidate models (teacher forcing:
the local model "re-reads" the answer token by token). The right candidate is the one under which the answers are
most likely.

| Family | Model selected | NLL per token | Next candidate |
|---|---|---|---|
| QWEN_B | Qwen2.5-1.5B-Instruct | 0.39 | 0.5B: 0.82 |
| QWEN_A | Qwen2.5-3B-Instruct | 0.35 | 1.5B: 0.74 |
| LLAMA | Llama-3.2-1B-Instruct | 0.63 | 3B: clearly worse |
| `ref` | SmolLM2-360M-Instruct | 0.94 | 1.7B: 1.41 |

**Why this is not brute force**: we choose among 2 to 4 candidates, with huge gaps, and **without any label or
score**. It is model selection by maximum likelihood. The MET paper itself assumes access to the reference model.
And the final ranking uses the **same 33 endpoints** (hidden pairs), so nothing has to generalise to other models.

### Step 4 — One test per kind of variant, validated on the `ref` endpoints

Once the base is known, there is a common yardstick for the 11 endpoints of a family. Every test was validated where
the truth is known before being applied:

| Variant | Test | On the `ref` endpoints | Statistical idea |
|---|---|---|---|
| fp8 | likelihood ratio Σ[log p_fp8(y) − log p_base(y)], fp8 simulated locally | ref-02: +19; the other 8: −41 to −225 | **Neyman–Pearson**: when both hypotheses are known, the likelihood ratio is the most powerful test |
| T = 1 | temperature that maximises the likelihood of the answers | ref-08: 1.004; plain: 0.715 | maximum-likelihood estimate of a decoding parameter |
| LoRA | correlation of token-frequency residuals (observed − expected under the base) | the 3 LoRAs correlate with each other (0.31 to 0.75), everything else < 0.2 | the three LoRAs are the same adapter at three strengths → same direction of deviation |
| watermark | rate of identical repeated answers | ref-06: 37 % versus 5 to 15 % | empirical signature: the watermark makes the output "stickier" |
| system prompt / persona | make the endpoint recite its instructions | ep-08 recites a system prompt; ep-16, ep-20, ep-23 a persona | direct evidence |

**The principle**: first **recognise and neutralise the cosmetics** (evidence per endpoint), only then look for
weight changes. What stays undetectable (LoRA 0.3, sometimes fp8 on the larger model) is **absorbed** into the base
group rather than guessed: isolating an endpoint wrongly costs more wrong pairs than leaving it in the group.

### Step 5 — Build the partition

Per family: one base group (the cosmetic variants + the base), the rest isolated. Working hypothesis "6 + 5"
(6 cosmetic, 5 weight variants), consistent with the `ref` endpoints and with the observed scores.

## 3. What is *not* method: using the leaderboard

Between 01:00 and 05:00 the team:

1. **submitted diagnostics**: two families "all different", the third one split, to measure family by family where
   points were lost (01:19, 01:50, 02:21);
2. **enumerated the structures compatible** with the 7 public scores obtained (`task2/src/structure/fit.py`);
3. **chose the next submission by a Bayesian computation** from those scores (`task2/src/structure/bayes.py`), with an
   automatic submission loop.

That uses the leaderboard as an **oracle**. It is allowed, but the statement warns that "a submission that is tuned to
the leaderboard can score high there and close to nothing in the final ranking".

## 4. How much of the 0.85 comes from evidence, how much from the leaderboard?

Partition at 0.85, endpoint by endpoint (see [`task2/results/endpoint_evidence.csv`](../../task2/results/endpoint_evidence.csv)):

| Family | Base group | Placed by evidence validated on `ref` | Placed by public scores | Judgement call (no decisive evidence) |
|---|---|---|---|---|
| QWEN_A | 01, 02, 04, 12, 23, 33 | 02 (system prompt), 23 (persona), 04 (watermark, repeats 0.36), 33 (T = 1) | black-box split of 22:45, **confirmed by the 01:19 diagnostic** (0.55, the value expected from a correct split) | — |
| QWEN_A | alone: 14, 17, 18, 24, 25 | 17, 18, 25 (correlated residuals, permutation MMD) | same (split confirmed) | — |
| QWEN_B | 03, 06, 07, 08, 20, 21, 30 | 08 (system prompt), 20 (persona), 06 (watermark, 0.25), 30 (T = 1) | **03, 07** (P = 1.00) | **21** (exactly one of {21, 32} is in the group, P = 0.50 each) |
| QWEN_B | alone: 22, 26, 28, 32 | 22 (fp8, likelihood ratio), 28 (permutation MMD, strong shift) | **26** (P = 0.31) | **32** |
| LLAMA | 10, 13, 16, 27, 29, 31 | 29 (system prompt: verbose preamble), 16 (persona), 10 (watermark, 0.20), 31 (T = 1) | **13** (P = 0.62) | **27** (P = 0.48) |
| LLAMA | alone: 05, 09, 11, 15, 19 | 05 and 11 (correlated residuals), 15 (fp8), 09 (LoRA dose-effect axis) | — | **19** (P = 0.57) |

25 endpoints out of 33 are placed by evidence or by a confirmed split, **4** by public scores (03, 07, 13,
26) and **4** by judgement (21, 32, 27, 19).

## 5. A point to watch: ep-28 and ep-29

A second, CPU-only white-box pass read these two endpoints differently from the final partition:

| Endpoint | CPU pass | Partition at 0.85 | Reading |
|---|---|---|---|
| ep-28 (QWEN_B) | T = 1 signature → cosmetic, base group | **alone** (strong shift in the MMD test) | real disagreement: the MMD test sees strong variants, but T = 1 also shifts the answers |
| ep-29 (LLAMA) | T = 1.0 and 27 % truncated answers → LoRA 1.5 signature | **base group** (verbose system prompt) | probably compatible: a verbose preamble also lengthens answers, which the CPU pass may have taken for a LoRA |

## 6. Risks for the final ranking

1. **The hidden pairs.** The final score uses the other pairs of the same endpoints. Correct evidence on an endpoint
   improves public **and** hidden pairs. A placement guessed from the public score may be right… or just lucky on the
   public half.
2. **The winner's curse.** After some twenty submissions differing by a few endpoints, the best one is partly the
   luckiest. Moving one endpoint touches only about 3 public pairs, and the score comes back rounded. Expect a final
   score somewhat below 0.85.
3. **The "6 + 5" hypothesis** is not proven for every family: it structures the judgement calls (19, 21, 27, 32).

## 7. Frequent questions

| Question | Short answer |
|---|---|
| "Isn't identifying the model cheating?" | No: no label, no score, 2 to 4 candidates, huge gaps. MET assumes a reference model. |
| "What if the server quantised differently?" | That is the main risk of the fp8 test; it was validated on ref-02, served by the same server. |
| "Why not the paper's MMD?" | Tried: it sees strong variants, not light ones, and it reacts to cosmetics. |
| "How many queries?" | About 3,256 out of 3,300; the white box costs none, it re-reads answers already paid for. |

See also: [reading notes on Model Equality Testing](../papers/model_equality_testing.md).
