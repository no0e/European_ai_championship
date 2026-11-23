# Task 3 — our method, explained and criticised

> Written during the event (22/11, around 10:50) from the log of the team's 92 submissions, with the same plan as
> the [task 2 note](task2_method.md): understand what we did, separate method from leaderboard tuning, be able to
> explain it. Nothing was re-run for this note. `src/…` paths refer to `task3/src/`. The task itself:
> [statement](../tasks/task3_unlearning.md) · [solution README](../../task3/README.md).

## 1. The problem in one sentence

We receive a ViT-B/16 (100 classes, trained from scratch) and 3,500 **unlabelled** images it must forget. We must
return weights that behave like **gold**: the average of several models retrained without those images. No "retain"
set, no validation set.

The score multiplies two terms:
- **forget** = `mean_std_score × prediction_score`, on the images to forget **and** on hidden altered copies (half
  and half):
  - `mean_std_score`: the mean **and the standard deviation** of the per-image loss must match gold's;
  - `prediction_score`: image by image, the predicted distribution (softmax at T = 2) must match gold's (KL);
- **retain** = accuracy × KL similarity to the original model on a hidden control set;
- and **0** if the hidden accuracy drops below 70 %.

## 2. The key idea: forgetting is not being wrong

**The naive mistake**: push the loss up on the images to forget (gradient ascent, SCRUB pushed hard). The model gets
them wrong, but **too much and too uniformly**. The SCRUB paper says it in its own way (§ 3.2, SCRUB+R): an
abnormally high error on forgotten examples makes them identifiable. A Streisand effect: an example forgotten "too
hard" stands out.

**What gold does**: it never saw these images, so for gold they are **test images**. It gets some right, others
wrong, with **large and widely spread** losses (around 5 ± 4.5 according to a model of the score fitted on seven
submissions, see § 5). Easy images stay right, hard ones become wrong.

**Example**: a very typical photo of a boat — gold recognises it without having seen it. An ambiguous photo of a
lantern — gold takes it for a figurine. The original model "learned both by heart". Forgetting well means making the
first one **a little less confident** and the second one **wrong**, not making both wrong.

## 3. The method, step by step

### 3.1 A simulator, to get a ground truth

On the real task, gold is only visible through a public score. To understand the problem without wasting
submissions, the team built a **simulator**: a ViT trained from scratch on CIFAR-100, the same forget-set structure
(45 classes, same counts), **4 retrained golds** and the exact scoring.

It plays the role of task 2's `ref` endpoints: a lab where the answer is known. Within an hour it gave the
**structure** of the method:
- softening the outputs is worth only 0.04 of forgetting (as on the leaderboard);
- setting the original-class margin according to difficulty is worth 0.37 to 0.46;
- KL distillation alone does not reproduce the tail of the losses.

Its **fine settings**, however, did not transfer (see § 4).

### 3.2 Build one target per image (`src/tg.py`)

1. **Original class** y = the original model's prediction. Labels are not given; the original model was trained on
   these images, so its prediction is very probably the right label.
2. **Difficulty** u(x) ∈ [0, 1]: the **rank** of the image by the original model's margin, averaged over the clean
   image and 4 altered views (0 = easy, 1 = hard).
3. **Imposed margin** of the original class (its logit minus the best of the others):

   m(x) = μ − σ · Φ⁻¹(u(x)),   with μ = −2.5 and σ = 7.

   **This is a quantile transform**: images are ranked by difficulty, then given the quantiles of a normal law
   N(μ, σ²). The imposed margins therefore follow exactly that law, in the order of difficulty.

   **Worked example**:
   - median image (u = 0.5): Φ⁻¹ = 0 → m = −2.5, slightly wrong;
   - hard image (u = 0.9): Φ⁻¹ ≈ 1.28 → m ≈ −11.5, clearly wrong;
   - easy image (u = 0.1): Φ⁻¹ ≈ −1.28 → m ≈ +6.5, still right (then halved by `--pos_scale 0.5`: images gold keeps
     right are less sure of themselves).

   **Implicit consequence**: P(m > 0) = Φ(μ/σ) = Φ(−0.36) ≈ 0.36. These settings act as if gold classified about
   **36 %** of the forgotten images correctly, like a test accuracy.
4. **The other logits** are the original model's, slightly spread apart (`tb = 1.4`). Only the original-class logit
   moves: the model chooses by itself which wrong class to go to, which resembles what gold would do.

### 3.3 Distil towards these targets (`src/k3.py distill`)

The student starts from the original model (30 epochs, lr 1e-4) and learns:
- **KL at T = 2 towards the targets** on the images to forget → aims at the `prediction_score`;
- **a Huber loss on the per-image cross-entropy** → aims at the `mean_std_score`. Without it the tail of the loss
  distribution is not reproduced, so the standard deviation is wrong;
- **altered copies** (crop, flip, colour, blur, noise) with a margin lowered by a further 4.5. Measured diagnostic:
  the naive student forgot **less** on the copies (loss 3.18 on a crop, 4.01 on a flip, versus 4.58 on the clean
  image), whereas gold rather loses more there. Fixed: 0.352 → 0.381;
- **5,000 COCO images as anchors**, on which the student must stay equal to the original model → aims at `retain`
  (KL similarity to the original).

### 3.4 Average the weights of six models (`src/k3.py avg`)

Six models from the same plateau of settings, all started from the original model, averaged parameter by parameter:
**0.470**, versus 0.459 to 0.465 for each alone.

**Why it works**: each training run has its own noise (±0.006, measured by re-running a setting with another seed).
The average removes that noise. It is also consistent with the target: **gold is itself an average of several
retrained models**, hence a smoothed output. (Averaging weights only makes sense because the six models start from
the same point and stay close to it; with twelve, 0.469: no further gain.)

## 4. What is method, what comes from the leaderboard

| Element | Origin | Solidity |
|---|---|---|
| "Forgetting = behaving at test time", rank-based margin | simulator + reading the scoring formula | **method** |
| Huber loss on the per-image CE | simulator (tail of the losses) | **method** |
| Lowered margin on altered copies | measured local diagnostic | **method** |
| COCO anchors | reading the scoring formula (retain term) | **method** |
| Weight averaging | measured training noise | **method** |
| Values μ = −2.5, σ = 7, tb = 1.4, view_delta 4.5, pos_scale 0.5, 30 epochs, 35 % strong views | **axis-by-axis search on the public score** | tuned on the leaderboard |
| "Gold's loss is around 5.2 ± 4.5" | model of the score fitted on 7 submissions | **inferred from the leaderboard** |

**The honest point**: the structure comes from reasoning and a simulator; the **fine settings** come from the
leaderboard. The simulator even pointed the other way for `tb` (squeeze the other logits), and the leaderboard
contradicted it (tb 1.0: 0.431; 1.4: 0.448). 92 submissions in total.

## 5. Is this leaderboard overfitting?

**Less than for task 2, and here is why.**

- **What is tuned is global.** About ten parameters applied to every image, not an image-by-image decision. Few
  degrees of freedom, so little room to fit the noise of the public part.
- **Public and final are drawn from the same population.** The statement: about 30 % of the scoring data for the
  public score, 70 % for the final ranking, split by a hash of each image's identifier (an altered copy follows its
  source image). A good global setting on 30 % is a good setting on 70 %.
- **Compared with task 2**: there, the leaderboard decided the place of **specific endpoints**, which can only be
  checked on the public half of the pairs.

**But two risks remain.**

1. **The winner's curse.** Among 92 submissions, the best is partly the luckiest. The gaps between the best ones
   (0.461, 0.465, 0.470) are of the same order as the training noise (±0.006). The noise of the **evaluation** on 30 %
   of the images was not measured and adds to it. Expect a somewhat different final score, probably a little lower.
2. **The weight average is the most robust choice**: it does not depend on the luckiest model, it gives the real level
   of the setting. A good argument for having submitted it last.

## 6. What did not work, and what it teaches

| Idea | Result | Lesson |
|---|---|---|
| Soften every output (T = 8) | 0.024 | gold is not "less sure everywhere", it is wrong on some images |
| SCRUB, extrapolated folds | 0.15, strong drift | pushing away from the teacher overshoots the target |
| "Gold never saw the 9 heavy classes" | 0.138 versus 0.346 | refuted: gold knows every class |
| Neighbour-based ranking | +30 % on the simulator, −9 % on the real task | a fine setting of the simulator does not transfer |
| Keep strong views equal to the original | 0.427 | any protection near the images to forget stops forgetting from reaching the copies, which count for half |
| Same wrong class for every copy of an image | 0.345 | the wrong class must be read on each copy |
| 50 epochs instead of 30 | 0.403 | fitting the targets better makes the rest drift: retain costs more than the fit brings |

## 7. In short

- **The core**: "Forgetting is not being wrong: it is behaving like a model that sees these images for the first time.
  We set a target margin for each image by a quantile transform of its difficulty rank, then distil, aiming at both
  the predicted distribution and the loss distribution, on the images and on their altered copies."
- **The statistician's angle**: quantile transform, one loss term per moment the scoring measures (mean and standard
  deviation), model averaging to reduce variance, a simulator to get a ground truth.
- **The honest part**: "The structure comes from the simulator; the fine settings come from the public score, a
  limited risk because they are global and public and final data come from the same population."

### Frequent questions

| Question | Short answer |
|---|---|
| "Why not SCRUB, the recommended paper?" | Tried: 0.15. It moves the student away from the teacher without a precise target; the score asks to **reproduce** gold, not to move away from the original. |
| "Without labels, how do you know the class?" | The original model's prediction: it was trained on these images, so it is very probably right. |
| "Any theoretical guarantee of forgetting?" | None; it is an imitation of gold measured by the score, as in SCRUB. A "certified removal" idea was tested: no effect here. |
| "What about the 70 % barrier?" | Far away: disabling 3 whole classes costs only 5 % of the score. |
| "Would it work on another model?" | The structure yes (validated on a different simulator); the fine settings would have to be redone. |

See also: [reading notes on SCRUB](../papers/scrub_unlearning.md).
