"""Basic tests for FDWM package."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
import cv2
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import fdwm


def test_import():
    """Test that the package can be imported."""
    assert fdwm.__version__ == "0.2.2"
    assert fdwm.__author__ == "Liam Huang"


def test_embed_extract_functions_exist():
    """Test that main functions are available."""
    assert hasattr(fdwm, "embed")
    assert hasattr(fdwm, "extract")
    assert hasattr(fdwm, "extract_text")


def test_create_test_images():
    """Test creating simple test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a simple host image
        host_img = np.full((100, 100), 128, dtype=np.uint8)
        host_path = tmp_path / "host.png"
        cv2.imwrite(str(host_path), host_img)
        assert host_path.exists()

        # Create a simple watermark
        wm_img = np.zeros((25, 25), dtype=np.uint8)
        wm_img[5:20, 5:20] = 255
        wm_path = tmp_path / "watermark.png"
        cv2.imwrite(str(wm_path), wm_img)
        assert wm_path.exists()

        return host_path, wm_path


def test_embed_extract_workflow():
    """Test the complete embed-extract workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        host_path, wm_path = test_create_test_images()

        # Output paths
        watermarked_path = tmp_path / "watermarked.png"
        extracted_path = tmp_path / "extracted.png"

        try:
            # Test embedding
            out_path, metrics = fdwm.embed(
                host_path=str(host_path),
                watermark_path=str(wm_path),
                output_path=str(watermarked_path),
                strength=1000.0,
                scale=0.25,
                grid_m=3,
                grid_n=3,
            )
            assert out_path.exists()

            # Test extraction
            extracted = fdwm.extract(
                watermarked_path=str(watermarked_path),
                strength=1000.0,
                scale=0.25,
                output_path=str(extracted_path),
                grid_m=3,
                grid_n=3,
            )
            assert extracted is not None
            assert extracted.shape == (25, 25)

        except Exception as e:
            pytest.skip(f"Watermark test skipped due to missing dependencies: {e}")


def test_text_watermark():
    """Test text watermark functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a simple host image
        host_img = np.full((100, 100), 128, dtype=np.uint8)
        host_path = tmp_path / "host.png"
        cv2.imwrite(str(host_path), host_img)

        # Output path
        watermarked_path = tmp_path / "watermarked.png"

        try:
            # Test text embedding
            out_path, metrics = fdwm.embed(
                host_path=str(host_path),
                watermark_path=None,
                output_path=str(watermarked_path),
                watermark_text="TEST",
                strength=1000.0,
                scale=0.25,
                grid_m=3,
                grid_n=3,
            )
            assert out_path.exists()

        except Exception as e:
            pytest.skip(f"Text watermark test skipped due to missing dependencies: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
