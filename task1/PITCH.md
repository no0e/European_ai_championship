# Task 1 — Where is the watermark? (2-minute pitch)

About 270 words, two minutes at a steady pace. Every figure comes from `README.md` in this folder.

**[The paper — 20 s]**
"Our starting point was Pixel Seal, from Meta. Its whole design goal is invisibility: the watermark is trained
against a discriminator that hunts for any visible trace, and a decoder reads 256 bits from the whole image. The
paper answers *is this image watermarked?* Our task asked a harder question: *where?* — on a 128-by-128 grid, for
Pixel Seal, TrustMark and StegaStamp."

**[An image is a function — 25 s]**
"So we wrote down how each image was made. Three of the four edits follow one formula:
y = M · embed(x) + (1 − M) · x. The whole image is watermarked, then blended with the original through a mask. The
fourth edit, outpainting, leaves a rectangle, and we set it aside: on TrustMark, decoding the exact crop re-reads
the message, so we searched crops. That gave 75 rectangles."

**[Dense detectors and their training data — 30 s]**
"The formula changes everything: with the official embedders, we can manufacture unlimited training images with
perfect masks. A segmentation network on raw pixels learned nothing — exactly as Pixel Seal intends. So each
detector starts from the method's own decoder, the only network that can see the watermark, with a dense head on
top. Its training data: composites built on images of the other two methods, with real-shaped masks, object shapes
and holes. First detector: 0.71."

**[The gap we found — 20 s]**
"Synthetic data still beat real data. Correlating the recomputed watermark with the images showed why: the images
had been watermarked on a frame up to 17 percent larger, then cropped. Once we reproduced that, a model trained on no
real image at all beat our fine-tuned ensemble on fully watermarked images."

**[Discipline — 15 s]**
"The public score covers only 30 percent of the test set, so every decision was taken on paired five-fold
cross-validation with bootstrap intervals, never on the leaderboard."

**[Close — 10 s]**
"From 0.45 to 0.787 public, 0.772 in cross-validation, and about 0.94 on fully watermarked StegaStamp images."

## If the jury asks

- **Rectangles for every method?** No, TrustMark only: re-reading the exact crop recovers the message for TrustMark
  (1.00) but only 0.66 for Pixel Seal (section 3).
- **"No real image at all"?** Measured on fully watermarked images: Pixel Seal 0.667 versus 0.647, TrustMark 0.657
  versus 0.634 for the previous ensemble fine-tuned on real images (section 3, "How the frame was found").
