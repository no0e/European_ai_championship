Watermark Localisation (Images) - starter kit and submission format
(Task description, data and scoring: see the participant document.)

STARTER KIT

  download_models.py         downloads the three watermark models. Run once.
  wmloc/watermarks.py        unified wrappers: embed(image, bits), logits(images), decode(images), features(images)
  wmloc/data.py              unzips and loads the dataset as PyTorch datasets
                             (train items also carry mask, mask128 and full_watermark)
  wmloc/utils.py             bit accuracy, the official mask downsampling, the official per-image mIoU and score
  submission_template.py     writes a valid submission (all-zeros baseline) and prints the local train score

  pip install -r requirements.txt
  python download_models.py
  python -c "from wmloc.data import unzip; unzip('train.zip', 'release'); unzip('test.zip', 'release')"
  python submission_template.py --data release --out submission.pt

SUBMISSION FORMAT

A .pt file containing a dictionary:

  torch.save({"image_names": names, "masks": masks}, "submission.pt")

  - names[i] is the image name (file name without .png) of masks[i]
  - masks: tensor of shape [1020, 128, 128], entries 0 or 1 (bool or uint8 recommended)
  - every test image exactly once, no duplicates, no other entries
  - file size at most 25 MB (a bool tensor is about 17 MB)

Files are loaded with torch.load(..., weights_only=True); anything other than plain tensors, lists, strings and
dicts is rejected. Invalid submissions are rejected automatically.

To submit, set BASE_URL, API_KEY and SUBMIT = True in submission_template.py; it uploads the written file to the
leaderboard server (submit(path) can also be called on its own).
