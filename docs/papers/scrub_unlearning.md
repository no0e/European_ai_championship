# SCRUB — Towards Unbounded Machine Unlearning — reading notes (task 3)

**M. Kurmanji, P. Triantafillou, J. Hayes, E. Triantafillou.** *Towards Unbounded Machine Unlearning.* NeurIPS 2023.
[arXiv:2302.09880](https://arxiv.org/abs/2302.09880) · code: [Meghdad92/SCRUB](https://github.com/Meghdad92/SCRUB).
Longer course, in French: [`../courses/scrub_unlearning_course_fr.pdf`](../courses/scrub_unlearning_course_fr.pdf);
toy implementation: [`../courses/toy/scrub_minimal.py`](../courses/toy/scrub_minimal.py).

## The question

Given a trained model and a forget set D_f, produce a model that behaves like one **retrained without D_f**, keeping
its performance on the retain set D_r — without paying for retraining. The paper's point is that what "forgetting
well" means depends on the application — removing biases, resolving confusion caused by mislabelled data, user
privacy — and that one method should work across them.

## The method

SCRUB is a **student–teacher** scheme. The teacher is the original model, frozen; the student starts as a copy.

    min over θ_u   α/N_r Σ_{D_r} KL(p_o ‖ p_u)  +  γ/N_r Σ_{D_r} CE(p_u, y)  −  1/N_f Σ_{D_f} KL(p_o ‖ p_u)

In practice the optimisation alternates:
- **max-steps** on D_f: move the student *away* from the teacher (maximise the KL);
- **min-steps** on D_r: keep the student *close* to the teacher and accurate (KL + cross-entropy);
- a few final min-steps alone, to repair the damage done by the max-steps.

**SCRUB+R (rewinding)**, for privacy: SCRUB aims at a high forget error, but an uncharacteristically high error on
the deleted examples makes them identifiable by membership-inference attacks. SCRUB+R measures, at the end of
training, the error on a held-out set from the same distribution as D_f — the error of a model that never saw these
examples — then rewinds to the checkpoint whose forget error is closest to it.

## Points worth knowing

- The student starts at a stationary point (student = teacher, so the KL and its gradient are zero): the first
  max-step does almost nothing; the min-steps move the student first.
- The max objective is unbounded: pushed too far, logits explode and the model becomes over-confident, itself a
  detectable signature.
- In selective unlearning the target is not 100 % error on D_f but **the error the model would have on unseen
  data** — usually much lower.

## What we used for task 3, and where we departed from it

- **What we kept**: the student–teacher distillation, starting from the original model; the idea that the target is
  the behaviour of a model that never saw the data, not maximal error (the same insight as SCRUB+R).
- **What did not fit**: SCRUB pushes the student *away* from the teacher without a precise target. The task scores
  how closely the submission **reproduces gold** — the mean and standard deviation of its per-image loss and its
  per-image output distribution — on the images and on altered copies. SCRUB folds extrapolated in weight space
  scored 0.15 and drifted strongly. There was also **no retain set** to run the min-steps on.
- **Our replacement**: build an explicit target per image — the original logits with only the original-class logit
  lowered, by a margin set by the image's difficulty rank through a quantile transform — and distil towards it, with
  a per-image cross-entropy term for the loss distribution, altered copies, and COCO images as anchors in place of
  the retain set. Then average the weights of six such models. 0.4700 public.
- Details: [`task3/README.md`](../../task3/README.md); method explained in French:
  [`../courses/task3_method_explained_fr.md`](../courses/task3_method_explained_fr.md).

## Other references given with the task

- V. S. Chundawat et al., *Zero-Shot Machine Unlearning*, IEEE TIFS 2023 — [arXiv:2201.05629](https://arxiv.org/abs/2201.05629).
- C. Fan et al., *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency*, ICLR 2024 —
  [arXiv:2310.12508](https://arxiv.org/abs/2310.12508).
- R. Zhang et al., *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*, COLM 2024 —
  [arXiv:2404.05868](https://arxiv.org/abs/2404.05868).
