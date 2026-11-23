# Task 1 — Localise an image watermark (official statement, summarised)

Supervisors: Adam Dziedzic and Franziska Boenisch. Task developed by Simon Hector, Michel Meintz, Louis Kerner and
Maitri Shah (CISPA / SprintML). Our solution: [`task1/README.md`](../../task1/README.md).

## Goal

Every image was watermarked with one of three public **post-hoc** watermarking methods — **PixelSeal**, **TrustMark**
or **StegaStamp** — in a rectangular subset of the original image (aspect ratio at most 16:1 for PixelSeal and
StegaStamp, 2:1 for TrustMark), then edited. For every test image, return a binary mask: 1 = watermarked region,
0 = clean region.

## Why it is hard

Methods such as Watermark Anything are trained for localisation: they say which pixels carry the watermark.
PixelSeal, TrustMark and StegaStamp are only trained to **decode one message from the whole image**. The question is
whether such watermarks can be localised anyway. The edits range from large (half the image) to small and irregular
(a thin object, a face), and the three methods behave very differently, so no single approach is likely to fit all.

Why it matters: without localisation, a later edit is attributed to the original watermark owner, which lets a third
party frame that owner by adding undesirable content to a watermarked image.

## The four kinds of edit

| Edit | Where the watermark is |
|---|---|
| Outpainting — a watermarked image extended into a larger one | the original image only, as a rectangle |
| Object removal / alteration | everywhere except a hole |
| Object addition — something unwatermarked pasted on top | everywhere except a hole |
| Object transplantation — a watermarked object pasted into a clean image | only on the object |

## Data

1,530 PNG images, 512×512, split equally across the three methods.

- **Train**: 510 images (170 per method), with the image, the ground-truth mask (512×512 and 128×128), the
  watermarking method and the embedded message.
- **Test**: 1,020 images (340 per method), with the image, the method and the embedded message.

The starter kit (Hugging Face dataset `SprintML/Hackathon_Watermark_Localisation_Task_Paris`) provides data loaders,
unified PyTorch wrappers for the three extractors **and embedders** (`decode(images) -> bits`, with access to
intermediate feature maps), bit accuracy, the exact mask downsampling, a local mIoU evaluation and a submission
template. Its code is shipped in [`task1/src/starter_kit/`](../../task1/src/starter_kit/).

## Metric

Masks are scored on a **128×128 grid** (one cell = 4×4 pixels); a ground-truth cell is 1 if at least half of its
pixels are watermarked. Per image, for each class c ∈ {0, 1}:

    IoU_c = |P_c ∩ G_c| / |P_c ∪ G_c|        (IoU_c = 1 if the union is empty)
    mIoU_image = (IoU_0 + IoU_1) / 2

The score is the mean of `mIoU_image` over the test images: every image counts equally, whatever the size of its
edit. Both false positives and false negatives enlarge the union, so both are penalised. Our exact re-implementation
is [`task1/src/t1_metric.py`](../../task1/src/t1_metric.py).

## Submission

A `.pt` file holding `{"image_names": names, "masks": masks}`, masks of shape `[num_images, 128, 128]` with entries
0 or 1, each name exactly once, at most 20 MB. **Public leaderboard on 30 % of the test images, final ranking on the
other 70 %.**

## References given with the task

- T. Souček et al., *Pixel Seal: Adversarial-only training for invisible image and video watermarking*, 2025 —
  [arXiv:2512.16874](https://arxiv.org/abs/2512.16874). Our notes: [papers/pixel_seal.md](../papers/pixel_seal.md).
- T. Bui, S. Agarwal, J. Collomosse, *TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution
  Images*, ICCV 2025 — [arXiv:2311.18297](https://arxiv.org/abs/2311.18297).
- M. Tancik, B. Mildenhall, R. Ng, *StegaStamp: Invisible Hyperlinks in Physical Photographs*, CVPR 2020 —
  [arXiv:1904.05343](https://arxiv.org/abs/1904.05343).
- T. Sander, P. Fernandez, A. Durmus, T. Furon, M. Douze, *Watermark Anything with Localized Messages*, ICLR 2025 —
  [arXiv:2411.07231](https://arxiv.org/abs/2411.07231).

## What the statement implied for us

- The kit gives the **exact embedders**. Combined with the four edit types, this means a watermarked region is
  `M · embed(x, m) + (1 − M) · x`: exact training pairs can be synthesised at will. This is the founding idea of our
  solution ([`task1/README.md`](../../task1/README.md) § 2).
- The message is given at test time, so a crop of the image can be re-decoded and checked against it — the basis of
  our TrustMark rectangle search.
- A constant prediction is capped at 0.5 per image, and small regions weigh as much as large ones: the decision rule
  has to adapt to the regime (whole image watermarked with holes, or a small area).
