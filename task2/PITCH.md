# Task 2 — two-minute pitch

About 300 words, two minutes at a steady pace.

We had 33 endpoints, and we had to say which ones serve exactly the same model. The catch is in the rules: a system
prompt, a persona, a watermark or a different temperature does not change the model; an fp8 quantisation or a LoRA
adapter does.

The obvious start was the Model Equality Testing paper: an MMD test on the answers. On the nine practice endpoints,
where we knew the truth, it failed both ways. It flagged the system prompt and the watermark as different models, and
it could not see the fp8 or the weak LoRA. What changes the text a lot doesn't change the model, and what changes the
model barely changes the text.

First, the families. We ignored what each model claims to be — "Claude" was just a persona — and used habits of the
base model instead: its reaction to a garbage prefix, its knowledge cutoff. A deliberate submission confirmed the
family part was perfect.

Black-box tests then plateaued at 0.60. The jump came from a detail of the paper: its test assumes you have the
reference model. So we found it. Re-scoring the answers we had already paid for, token by token, identified the exact
checkpoints — Qwen2.5 3B and 1.5B, Llama-3.2 1B — by a wide margin. With the weights in hand, we could test precise
hypotheses. We fitted each endpoint's temperature, spotted watermarks from identical repeated answers, and caught
LoRAs from correlated token residuals. We caught fp8 with a likelihood ratio against our own simulated fp8: plus 19 on
the quantised practice endpoint, minus 40 or worse on all the others. Every test was validated on the practice
endpoints first, and none cost an extra query.

That took us to 0.85. Twenty-five of the 33 endpoints are placed by that evidence; for the last few, we used public
scores, and we say so.

---

Numbers behind the pitch: README section 2 (evidence per endpoint, `results/endpoint_evidence.csv`), section 3
(timeline) and "Why the white box unlocked the task" (checkpoints, fp8 likelihood ratio on the practice endpoints).
