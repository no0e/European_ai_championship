# Our methods, explained

Long-form notes written during the event to understand our own solutions, separate what is a method from what was
tuned on the leaderboard, and explain both clearly.

| Note | What it covers |
|---|---|
| [Task 2 — model equality testing](task2_method.md) | the trap of cosmetic disguises, identifying the checkpoint by likelihood, one test per kind of variant and the statistical idea behind each, which endpoints rest on evidence and which on public scores, risks for the final ranking |
| [Task 3 — unlearning](task3_method.md) | why forgetting is not being wrong, the simulator, rank-based targets as a quantile transform (with a worked example), distillation, weight averaging, what is method and what is tuning |

Task 1 is explained step by step, with figures, in its [README](../../task1/README.md).

## Toy scripts (CPU, about a minute each)

Small, self-contained versions of the ideas in two of the recommended papers, written while preparing.

| Script | What it shows |
|---|---|
| [toy/met_toy.py](toy/met_toy.py) | Model Equality Testing: MMD with the Hamming kernel and its permutation p-value, on a toy bigram "LLM", against temperature changes and a simulated quantisation (numpy only) |
| [toy/scrub_minimal.py](toy/scrub_minimal.py) | SCRUB and SCRUB+R on a small synthetic dataset, compared with a retrained oracle, with a basic membership-inference attack (torch, scikit-learn) |

```bash
python docs/explanations/toy/met_toy.py
python docs/explanations/toy/scrub_minimal.py
```
