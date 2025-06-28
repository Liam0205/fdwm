import subprocess
from pathlib import Path
import tempfile
import cv2
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import fdwm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gen_host(path: Path):
    img = np.full((256, 256), 180, dtype=np.uint8)
    cv2.putText(img, "HOST", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)
    cv2.imwrite(str(path), img)


def test_cli_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img1 = tmp / "a.png"
        img2 = tmp / "sub" / "b.png"
        img2.parent.mkdir()
        _gen_host(img1)
        _gen_host(img2)

        # run CLI with text watermark
        cmd = [
            sys.executable,
            "-m",
            "fdwm",
            "embed",
            str(tmp),
            "--watermark-text",
            "CLI DEMO",
            "--strength",
            "10000",
            "--scale",
            "0.25",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "Watermark applied" in res.stdout

        # verify images overwritten (simple mean difference)
        img_after = cv2.imread(str(img1), 0)
        assert (img_after.astype(int) - 180).mean() != 0, "Image not modified"


def test_cli_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        host = tmp / "h.png"
        _gen_host(host)
        # create small watermark image
        wm = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(wm, "WM", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        wm_path = tmp / "wm.png"
        cv2.imwrite(str(wm_path), wm)

        cmd = [
            sys.executable,
            "-m",
            "fdwm",
            "embed",
            str(host),
            "--watermark-img",
            str(wm_path),
            "--strength",
            "10000",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        assert "Watermark applied" in res.stdout

        img_after = cv2.imread(str(host), 0)
        assert (img_after.astype(int) - 180).mean() != 0


def test_cli_extract():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        host = tmp / "orig.png"
        _gen_host(host)

        # embed via API for setup
        fdwm.embed(
            host_path=str(host),
            watermark_text="EXTRACT DEMO",
            watermark_path=None,
            output_path=str(host),
            strength=20000.0,
            scale=0.25,
        )

        # now extract via CLI to image
        cmd = [
            sys.executable,
            "-m",
            "fdwm",
            "extract",
            str(host),
            "--output",
            str(tmp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        extracted_img = tmp / "orig_wm.png"
        assert extracted_img.exists()

        # extract text via CLI
        cmd_text = [
            sys.executable,
            "-m",
            "fdwm",
            "extract",
            str(host),
            "--text",
        ]
        res2 = subprocess.run(cmd_text, capture_output=True, text=True)
        assert res2.returncode == 0, res2.stderr
        assert "EXTRACT" in res2.stdout or "DEMO" in res2.stdout


if __name__ == "__main__":
    test_cli_text()
    test_cli_image()
    test_cli_extract()
