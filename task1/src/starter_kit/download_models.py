"""Download the three watermark models into ./models (or --models_dir).
  - PixelSeal: clones github.com/facebookresearch/videoseal at a pinned commit and downloads the PixelSeal
    checkpoint; writes models/pixelseal_card.yaml with absolute paths.
  - TrustMark: `pip install trustmark==0.8.0` downloads its weights on first use (nothing to do here).
  - StegaStamp: downloads the WAVES ONNX export and writes a copy that also outputs the decoder logits.
Needs: git, `pip install onnx` (only for the StegaStamp step)."""
import argparse, subprocess, urllib.request
from pathlib import Path

VIDEOSEAL_COMMIT = "b1bf4c0"
PIXELSEAL_CKPT = "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth"
STEGASTAMP_ONNX = "https://github.com/umd-huang-lab/WAVES/raw/main/decoders/stega_stamp.onnx"


def fetch(url, dst: Path):
    if dst.exists():
        print("exists:", dst); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url); urllib.request.urlretrieve(url, dst.with_suffix(".part")); dst.with_suffix(".part").rename(dst)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models_dir", default=str(Path(__file__).resolve().parent / "models"))
    d = Path(ap.parse_args().models_dir).resolve(); d.mkdir(parents=True, exist_ok=True)
    repo = d / "videoseal"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/videoseal.git", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", VIDEOSEAL_COMMIT], check=True)
    fetch(PIXELSEAL_CKPT, d / "pixelseal" / "checkpoint.pth")
    card = (repo / "videoseal" / "cards" / "pixelseal.yaml").read_text()
    card = card.replace(PIXELSEAL_CKPT, str(d / "pixelseal" / "checkpoint.pth"))
    card = card.replace("attenuation_config: configs/", f"attenuation_config: {repo}/configs/")
    (d / "pixelseal_card.yaml").write_text(card)
    fetch(STEGASTAMP_ONNX, d / "stegastamp" / "stega_stamp.onnx")
    out = d / "stegastamp" / "stega_stamp_logits.onnx"
    if not out.exists():
        import onnx
        m = onnx.load(str(d / "stegastamp" / "stega_stamp.onnx"))
        m.graph.output.append(onnx.helper.make_tensor_value_info("gen_decoder/decoder_out/BiasAdd:0", onnx.TensorProto.FLOAT, None))
        onnx.save(m, str(out))
    print("models ready in", d)


if __name__ == "__main__":
    main()
