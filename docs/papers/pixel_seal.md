# Pixel Seal — reading notes (task 1)

**T. Souček, P. Fernandez, H. Elsahar, S.-A. Rebuffi, V. Lacatusu, T. Tran, T. Sander, A. Mourachko.**
*Pixel Seal: Adversarial-only training for invisible image and video watermarking.* Meta FAIR, December 2025.
[arXiv:2512.16874](https://arxiv.org/abs/2512.16874) · code: [facebookresearch/videoseal](https://github.com/facebookresearch/videoseal)
(MIT). Longer course, in French: [`../courses/pixel_seal_course_fr.pdf`](../courses/pixel_seal_course_fr.pdf).

## In one paragraph

Pixel Seal is a **post-hoc** watermark (applied after an image is generated, not during generation) that hides a
**256-bit** message in an image or a video. An embedder adds an invisible residual; an extractor reads the message
back, even after crops, resizes, JPEG and colour changes. Its claim is that imperceptibility should be learned
**only through an adversarial discriminator**, with no pixel-wise perceptual loss, and that this needs a careful
three-stage training schedule.

## The model

- **Embedder**: a U-Net (≈ 44 M parameters) working on the luminance channel Y. **Extractor**: a ConvNeXt-V2 Tiny
  (≈ 33 M). Both run at 256×256.
- **Embedding**: `x_w = x + α · (resize↑(w) ⊙ JND(x))`, with `w = Emb(resize↓(x), m)`. The residual is computed at
  low resolution, upsampled, and modulated by a **just-noticeable-difference** map computed at native resolution:
  strong in textured areas, weak in flat ones. The watermark hides in the texture.
- **Detection without calibration**: the number of matching bits gives a binomial p-value,
  `p = Σ_{k ≤ d_H} C(n, k) / 2^n`, reported as −log10 p.

## The ideas

1. **Adversarial-only loss.** Only a message loss (BCE on the bits) plus a patch discriminator with a hinge loss; MSE
   and LPIPS are dropped. Pixel-wise losses penalise every change equally, whereas a discriminator only penalises what
   looks unnatural.
2. **Three-stage schedule.** (i) Message loss alone at a strong watermark (α = 1) until bit accuracy saturates;
   (ii) add the adversarial loss and decay α from 1.0 to 0.2 with a cosine; (iii) fine-tune at α = 0.2. Starting
   weak, or adversarial from the start, learns nothing (50 % bit accuracy).
3. **High-resolution adaptation.** Training with random resolutions and with the JND map and augmentations applied at
   native resolution before downsampling ("inference simulation"); non-differentiable augmentations (JPEG) through a
   straight-through estimator.
4. **Video** (a fourth, inference-only idea): temporal average pooling inside the embedder, about 1.7× faster.

## Results and limits stated by the paper

- Best on geometric attacks (crop, resize, rotation); competitive on combined attacks; weaker, like every method, on
  H.264/HEVC video compression. Image quality around 49 dB PSNR.
- The discriminator trades visibility against robustness; **no analysis against adaptive adversaries, removal or
  forgery**.

## What we used for task 1

- The task gives the **official PixelSeal embedder and extractor**. Since the watermarked region of an edited image
  is `M · embed(x, m) + (1 − M) · x`, we generated as many exact training composites as we wanted.
- The paper's architecture told us where the signal lives: the **extractor itself** is the right backbone for a
  dense "watermarked / not watermarked" detector. A generic ImageNet backbone learns nothing on PixelSeal (AUC 0.5);
  the PixelSeal ConvNeXt-V2 does.
- The JND modulation explains why the signal is weak in flat areas: "no signal" is not the same as "clean", which is
  why the decision rule is chosen per method and per regime, out of fold.
- Details: [`task1/README.md`](../../task1/README.md) § 2.

## The other two methods of task 1

- **TrustMark** (Bui, Agarwal, Collomosse, ICCV 2025, [arXiv:2311.18297](https://arxiv.org/abs/2311.18297)): 100
  bits, residual generated at low resolution then upscaled, ResNet-50 decoder; decodes almost perfectly from an exact
  crop, which is what our rectangle search exploits.
- **StegaStamp** (Tancik, Mildenhall, Ng, CVPR 2020, [arXiv:1904.05343](https://arxiv.org/abs/1904.05343)): 100 bits,
  message → dense layer → 50×50×3 block pattern upscaled; the original paper already trains a segmentation network to
  find the encoded area.
- **Watermark Anything** (Sander, Fernandez et al., ICLR 2025, [arXiv:2411.07231](https://arxiv.org/abs/2411.07231)):
  the localised counterpart, whose extractor predicts presence and bits for every pixel — what the task asks us to
  recover for methods that were not trained for it.
