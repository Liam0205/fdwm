from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import fdwm


def generate_host_image(path: str, size: int = 512) -> None:
    """Generate simple grayscale host image and save to disk."""
    img = np.full((size, size), 200, dtype=np.uint8)  # light gray background
    cv2.putText(
        img,
        "HOST",
        (size // 8, size // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        30,
        5,
    )
    cv2.imwrite(path, img)


def generate_watermark(path: str, size: int = 128) -> None:
    """Generate simple black background white text watermark image and save to disk."""
    wm = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(
        wm,
        "WM",
        (10, size - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        255,
        5,
    )
    cv2.imwrite(path, wm)


def test_embed_extract():
    """Complete workflow test: embed -> extract -> calculate correlation coefficient."""
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    host_path = tmp_dir / "host.png"
    wm_path = tmp_dir / "wm.png"
    watermarked_path = tmp_dir / "host_wm.png"
    extracted_path = tmp_dir / "extracted.png"

    generate_host_image(str(host_path))
    generate_watermark(str(wm_path))

    # embed
    fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=5000.0,
        scale=0.25,
    )

    # extract
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=5000.0,
        scale=0.25,
        output_path=str(extracted_path),
    )

    # resize original watermark to same size
    wm_original = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
    wm_resized = cv2.resize(
        wm_original, extracted.shape[::-1], interpolation=cv2.INTER_AREA
    )

    # calculate Pearson correlation coefficient
    corr = np.corrcoef(wm_resized.flatten(), extracted.flatten())[0, 1]
    print(
        f"Correlation coefficient between original and extracted watermark: {corr:.3f}"
    )

    assert corr > 0.5, "Watermark extraction correlation too low, possible issue."
    print("✅ Library functionality test passed!")


def test_embed_extract_all_regions():
    tmp_dir = Path("tmp_all_regions")
    tmp_dir.mkdir(exist_ok=True)
    host_path = tmp_dir / "host.png"
    wm_path = tmp_dir / "wm.png"
    watermarked_path = tmp_dir / "host_wm.png"
    extracted_path = tmp_dir / "extracted.png"
    generate_host_image(str(host_path))
    generate_watermark(str(wm_path))
    strength = 5000.0
    scale = 0.25
    # Test corners
    fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        region_type="corners",
    )
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        output_path=str(extracted_path),
        region_type="corners",
    )
    assert extracted.shape == (int(512 * scale), int(512 * scale))
    # Test center
    fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        region_type="center",
    )
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        output_path=str(extracted_path),
        region_type="center",
    )
    assert extracted.shape == (int(512 * scale), int(512 * scale))
    # Test random
    fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        region_type="random",
        random_seed=123,
    )
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        output_path=str(extracted_path),
        region_type="random",
        random_seed=123,
    )
    assert extracted.shape == (int(512 * scale), int(512 * scale))
    # Test error: missing random_seed (should use default 42, so no error)
    fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        region_type="random",
    )
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=strength,
        scale=scale,
        output_path=str(extracted_path),
        region_type="random",
    )
    assert extracted.shape == (int(512 * scale), int(512 * scale))
    # Test error: invalid region_type
    try:
        fdwm.embed(
            host_path=str(host_path),
            watermark_path=str(wm_path),
            output_path=str(watermarked_path),
            strength=strength,
            scale=scale,
            region_type="invalid",
        )
    except ValueError as e:
        assert "Invalid region_type" in str(e)
    else:
        assert False, "Expected ValueError for invalid region_type"


if __name__ == "__main__":
    test_embed_extract()
    test_embed_extract_all_regions()
