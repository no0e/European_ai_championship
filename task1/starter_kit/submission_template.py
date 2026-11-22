"""Template: predict a 128 x 128 mask for every test image, write the submission file and (optionally) submit it.
    python submission_template.py --data release --out submission.pt
Format: torch.save({"image_names": [str, ...], "masks": uint8/bool tensor [N, 128, 128] with 0/1}, path).
1 = watermarked, 0 = not watermarked. Every test image exactly once.
To submit, set API_KEY and SUBMIT = True below (or run submit("submission.pt") yourself)."""
import argparse
import os
import sys

import numpy as np
import requests
import torch

from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, score

def _team_api_key() -> str:
    """API_KEY from the environment, else from the nearest .env up the tree (repo root .env, git-ignored)."""
    import os
    from pathlib import Path
    if os.environ.get("API_KEY", "").strip():
        return os.environ["API_KEY"].strip()
    for d in Path(__file__).resolve().parents:
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() == "API_KEY" and v.strip():
                    return v.strip().strip("\"'")
    return "YOUR_API_KEY_HERE"


BASE_URL = "http://SERVER_IP"
API_KEY = _team_api_key()                # team key: export API_KEY=... or put it in the repo-root .env

TASK_ID = "34-watermark-localisation-image"
SUBMIT = False          # set True to upload the written file to the leaderboard


def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)


def submit(file_path):
    if not os.path.isfile(file_path):
        die(f"File not found: {file_path}")
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)


def predict(item, watermarks=None) -> np.ndarray:
    """Replace this. Must return a 128 x 128 array of 0/1 for one image.
    Available: item["image"] (512x512x3 uint8), item["method"], item["message"] (bits),
    and the watermark wrappers (wmloc.watermarks.load_watermark)."""
    return np.zeros((128, 128), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="submission.pt")
    a = ap.parse_args()
    train = LocalisationDataset(a.data, "train")
    print("local train score:", score([predict(it) for it in train], [it["mask128"] for it in train]))
    test = LocalisationDataset(a.data, "test")
    names, masks = [], []
    for it in test:
        p = np.asarray(predict(it), dtype=np.uint8)
        assert p.shape == (128, 128) and set(np.unique(p)) <= {0, 1}
        names.append(it["image_name"]); masks.append(p)
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.out)
    print("wrote", a.out, len(names), "masks")
    if SUBMIT:
        submit(a.out)


if __name__ == "__main__":
    main()
