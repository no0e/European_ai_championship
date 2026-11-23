# Research papers

Before the event, the organisers asked every participant to read three papers. Each one turned out to be the starting
point of one task.

| Task | Paper | Our reading notes | Paper |
|---|---|---|---|
| 1 — Watermark localisation | T. Souček, P. Fernandez et al., *Pixel Seal: Adversarial-only training for invisible image and video watermarking*, Meta FAIR, 2025 | [pixel_seal.md](pixel_seal.md) | [arXiv:2512.16874](https://arxiv.org/abs/2512.16874) |
| 2 — Model equality testing | I. Gao, P. Liang, C. Guestrin, *Model Equality Testing: Which Model Is This API Serving?*, ICLR 2025 | [model_equality_testing.md](model_equality_testing.md) | [arXiv:2410.20247](https://arxiv.org/abs/2410.20247) · [PDF here](gao2025_model_equality_testing.pdf) |
| 3 — Unlearning | M. Kurmanji, P. Triantafillou, J. Hayes, E. Triantafillou, *Towards Unbounded Machine Unlearning* (SCRUB), NeurIPS 2023 | [scrub_unlearning.md](scrub_unlearning.md) | [arXiv:2302.09880](https://arxiv.org/abs/2302.09880) |

Each note gives the paper's question, its method, its key results and limits, and then **what we took from it and
where we had to depart from it** for the task. The task statements cite further references; they are listed at the
end of each note and in [`../tasks/`](../tasks/).

**Copies of the papers.** *Model Equality Testing* is distributed under the Creative Commons Attribution 4.0 licence
(CC BY 4.0), so its PDF is included here, unmodified (© the authors). For the other two papers we only link to arXiv,
where the authoritative versions are.

## In one line each, and how they relate to our solutions

- **Pixel Seal** hides 256 bits with an embedder trained only against a discriminator. *We used its extractor as the
  backbone of a dense detector, and its official embedder to synthesise exact training composites.*
- **Model Equality Testing** turns "which model does this API serve?" into an MMD two-sample test with a Hamming
  kernel. *We kept the hypothesis-testing framing, but cosmetic disguises dominate the text, so we first identified
  the checkpoint by likelihood and then ran one test per kind of weight variant.*
- **SCRUB** unlearns by pushing a student away from its teacher on the forget set while holding it close elsewhere,
  and rewinds to avoid over-forgetting. *The task scores how well gold is imitated, so we gave the student an explicit
  per-image target instead of a direction to move away.*
