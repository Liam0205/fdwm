from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import fdwm

kStrength = 50000.0


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


def test_visual_invisibility():
    """Test that watermarking causes minimal visual difference in host image."""
    tmp_dir = Path("tmp_visual")
    tmp_dir.mkdir(exist_ok=True)

    host_path = tmp_dir / "host.png"
    wm_path = tmp_dir / "wm.png"
    watermarked_path = tmp_dir / "host_wm.png"

    generate_host_image(str(host_path))
    generate_watermark(str(wm_path))

    # Read original host image
    original = cv2.imread(str(host_path), cv2.IMREAD_GRAYSCALE)

    # Embed watermark
    out_path, metrics = fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=kStrength,
        scale=0.25,
        grid_m=3,
        grid_n=3,
    )

    # Read watermarked image
    watermarked = cv2.imread(str(watermarked_path), cv2.IMREAD_GRAYSCALE)

    # Calculate overall difference
    diff = np.abs(watermarked.astype(int) - original.astype(int))
    mean_diff = np.mean(diff)
    p90_diff = np.percentile(diff, 90)

    print(f"Mean pixel difference: {mean_diff:.2f}")
    print(f"90th percentile pixel difference: {p90_diff:.2f}")

    # Test 1: Overall visual similarity (mean difference should be small)
    assert mean_diff < 5.0, f"Mean difference too high: {mean_diff:.2f}"

    # Test 2: No extreme pixel changes (90th percentile difference should be reasonable)
    # Allow higher p90 difference as some pixels may have larger changes
    assert p90_diff < 15, f"90th percentile difference too high: {p90_diff:.2f}"

    # Test 3: PSNR (Peak Signal-to-Noise Ratio) should be high
    mse = np.mean((watermarked.astype(float) - original.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    print(f"PSNR: {psnr:.2f} dB")
    assert psnr > 25.0, f"PSNR too low: {psnr:.2f} dB"

    # Test 4: Most pixels should have minimal change
    pixels_with_small_change = np.sum(diff < 10)
    total_pixels = diff.size
    small_change_ratio = pixels_with_small_change / total_pixels
    print(f"Pixels with change < 10: {small_change_ratio:.1%}")
    assert (
        small_change_ratio > 0.8
    ), f"Too many pixels have large changes: {small_change_ratio:.1%}"

    print("✅ Visual invisibility test passed!")


def test_watermark_extraction_quality():
    """Test that extracted watermark is highly similar to original."""
    tmp_dir = Path("tmp_extraction")
    tmp_dir.mkdir(exist_ok=True)

    host_path = tmp_dir / "host.png"
    wm_path = tmp_dir / "wm.png"
    watermarked_path = tmp_dir / "host_wm.png"
    extracted_path = tmp_dir / "extracted.png"

    generate_host_image(str(host_path))
    generate_watermark(str(wm_path))

    # Embed watermark
    out_path, metrics = fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=kStrength,
        scale=0.25,
        grid_m=3,
        grid_n=3,
    )

    # Extract watermark
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=kStrength,
        scale=0.25,
        output_path=str(extracted_path),
        grid_m=3,
        grid_n=3,
    )

    # Read original watermark and resize to match extracted size
    wm_original = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
    wm_resized = cv2.resize(
        wm_original, extracted.shape[::-1], interpolation=cv2.INTER_AREA
    )

    # Calculate multiple similarity metrics
    # 1. Pearson correlation coefficient
    corr = np.corrcoef(wm_resized.flatten(), extracted.flatten())[0, 1]

    # 2. Structural Similarity Index (SSIM-like)
    def calculate_ssim(img1, img2):
        """Calculate a simplified SSIM-like metric."""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2)
        )
        return ssim

    ssim = calculate_ssim(wm_resized, extracted)

    print(f"Correlation coefficient: {corr:.3f}")
    print(f"SSIM-like metric: {ssim:.3f}")

    # Assertions for high quality extraction
    assert corr > 0.7, f"Correlation too low: {corr:.3f}"
    assert ssim > 0.7, f"SSIM too low: {ssim:.3f}"

    print("✅ Watermark extraction quality test passed!")


def test_embed_extract_basic():
    """Test basic embed and extract functionality."""
    tmp_dir = Path("tmp_basic")
    tmp_dir.mkdir(exist_ok=True)
    host_path = tmp_dir / "host.png"
    wm_path = tmp_dir / "wm.png"
    watermarked_path = tmp_dir / "host_wm.png"
    extracted_path = tmp_dir / "extracted.png"
    generate_host_image(str(host_path))
    generate_watermark(str(wm_path))
    scale = 0.25

    # Test basic embed and extract
    out_path, metrics = fdwm.embed(
        host_path=str(host_path),
        watermark_path=str(wm_path),
        output_path=str(watermarked_path),
        strength=kStrength,
        scale=scale,
        grid_m=3,
        grid_n=3,
    )
    extracted = fdwm.extract(
        watermarked_path=str(watermarked_path),
        strength=kStrength,
        scale=scale,
        output_path=str(extracted_path),
        grid_m=3,
        grid_n=3,
    )
    assert extracted.shape == (int((512 // 3) * scale), int((512 // 3) * scale))
    print("✅ Basic embed and extract test passed!")


if __name__ == "__main__":
    test_visual_invisibility()
    test_watermark_extraction_quality()
    test_embed_extract_basic()
