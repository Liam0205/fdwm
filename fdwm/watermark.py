import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pytesseract
import tempfile
import random


def _read_image(path: str | Path, gray: bool = True) -> np.ndarray:
    """Read image from *path* using OpenCV.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to image file.
    gray : bool, default ``True``
        If ``True``, image will be loaded as single-channel grayscale.

    Returns
    -------
    numpy.ndarray
        Image matrix.
    """
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Failed to load image file: {path}")
    return img


def _text_to_image(
    text: str,
    target_size: Tuple[int, int],
    *,
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
) -> np.ndarray:
    """Render *text* to a grayscale numpy array of *target_size* (rows, cols)."""
    rows, cols = target_size

    # If font_path not specified, try DejaVuSans.ttf (Pillow built-in, multi-language support)
    fallback_font_path = None if font_path else "DejaVuSans.ttf"

    # Initial font size estimate: try to fill height
    if font_size is None:
        font_size = rows  # will decrease until fit

    # Decrease font size to fit
    while font_size > 10:
        try:
            if font_path is not None:
                font = ImageFont.truetype(font_path, font_size)
            else:
                try:
                    font = (
                        ImageFont.truetype(fallback_font_path, font_size)
                        if fallback_font_path
                        else ImageFont.load_default()
                    )
                except Exception:
                    font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Simple line wrapping: split by character count
        avg_char_width = font.getlength("汉a") / 2  # approximate average width
        max_chars_per_line = max(1, int(cols / avg_char_width))
        wrapped = textwrap.fill(text, width=max_chars_per_line)

        dummy_img = Image.new("L", (cols, rows), color=0)
        draw = ImageDraw.Draw(dummy_img)
        text_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        if text_w <= cols and text_h <= rows:
            # Text fits
            img = Image.new("L", (cols, rows), color=0)
            draw = ImageDraw.Draw(img)
            # Center draw
            x = (cols - text_w) // 2
            y = (rows - text_h) // 2
            draw.multiline_text(
                (x, y), wrapped, fill=255, font=font, spacing=4, align="center"
            )
            return np.array(img)

        font_size -= 2  # font too large, decrease and retry

    # Minimum font size still doesn't fit, scale directly
    font = ImageFont.load_default()
    img = Image.new("L", (cols, rows), color=0)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((0, 0), text, fill=255, font=font)
    return np.array(img)


def _embed_center_region(host_dft_shift, watermark_norm, strength, cx, cy, region_size):
    """Embed watermark in the center region of the frequency domain."""
    rows, cols = host_dft_shift.shape
    wm_rows, wm_cols = watermark_norm.shape

    # Calculate center region boundaries
    start_row = max(0, cx - region_size // 2)
    end_row = min(rows, cx + region_size // 2)
    start_col = max(0, cy - region_size // 2)
    end_col = min(cols, cy + region_size // 2)

    # Resize watermark to fit center region
    center_rows = end_row - start_row
    center_cols = end_col - start_col
    wm_resized = cv2.resize(
        watermark_norm, (center_cols, center_rows), interpolation=cv2.INTER_AREA
    )

    # Embed in center region
    host_dft_shift[start_row:end_row, start_col:end_col] += strength * wm_resized


def _embed_random_regions(
    host_dft_shift, watermark_norm, strength, seed, num_regions=5
):
    """Embed watermark in random regions using specified seed."""
    rows, cols = host_dft_shift.shape
    wm_rows, wm_cols = watermark_norm.shape

    # Set random seed for reproducible embedding
    random.seed(seed)
    np.random.seed(seed)

    # Generate random positions for embedding
    for i in range(num_regions):
        # Random position within valid bounds
        start_row = random.randint(0, rows - wm_rows)
        start_col = random.randint(0, cols - wm_cols)

        # Random strength variation
        region_strength = strength * random.uniform(0.3, 1.0)

        # Random transformation (flip, rotate, etc.)
        wm_transformed = watermark_norm.copy()
        if random.choice([True, False]):
            wm_transformed = np.fliplr(wm_transformed)
        if random.choice([True, False]):
            wm_transformed = np.flipud(wm_transformed)

        # Embed in random region
        host_dft_shift[
            start_row : start_row + wm_rows, start_col : start_col + wm_cols
        ] += (region_strength * wm_transformed)


def embed(
    host_path: str | Path,
    watermark_path: Optional[str | Path],
    output_path: str | Path,
    *,
    watermark_text: Optional[str] = None,
    strength: float = 10.0,
    scale: float = 0.25,
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    region_type: Literal["corners", "center", "random"] = "corners",
    random_seed: Optional[int] = 42,
) -> Path:
    """Embed watermark in frequency domain with selectable regions.

    Parameters
    ----------
    host_path : str | Path
        Path to host image.
    watermark_path : str | Path, optional
        Path to watermark image.
    output_path : str | Path
        Path for output watermarked image.
    watermark_text : str, optional
        Text to embed as watermark.
    strength : float, default 10.0
        Embedding strength.
    scale : float, default 0.25
        Watermark scale relative to host image.
    font_path : str, optional
        Path to font file for text watermark.
    font_size : int, optional
        Font size for text watermark.
    region_type : {"corners", "center", "random"}, default "corners"
        Type of regions to embed watermark:
        - "corners": Embed in four corners (top-left, top-right, bottom-left, bottom-right)
        - "center": Embed in center region
        - "random": Embed in random regions using seed
    random_seed : int, optional
        Random seed for reproducible random embedding (required when region_type="random").
    """
    host = _read_image(host_path, gray=True)
    rows, cols = host.shape
    wm_rows = int(rows * scale)
    wm_cols = int(cols * scale)

    if watermark_text is not None:
        watermark = _text_to_image(
            watermark_text, (wm_rows, wm_cols), font_path=font_path, font_size=font_size
        )
    else:
        if watermark_path is None:
            raise ValueError("Must provide either watermark_path or watermark_text.")
        watermark_src = _read_image(watermark_path, gray=True)
        watermark = cv2.resize(
            watermark_src, (wm_cols, wm_rows), interpolation=cv2.INTER_AREA
        )

    watermark_norm = watermark.astype(np.float32) / 255.0
    host_dft = np.fft.fft2(host)
    host_dft_shift = np.fft.fftshift(host_dft)

    if region_type == "corners":
        # Embed in four corners
        # 1. Top-left corner (main region, strong signal)
        host_dft_shift[0:wm_rows, 0:wm_cols] += strength * watermark_norm

        # 2. Bottom-right corner (flipped, weak signal)
        host_dft_shift[rows - wm_rows : rows, cols - wm_cols : cols] += (
            0.5 * strength
        ) * np.flipud(np.fliplr(watermark_norm))

        # 3. Top-right corner (weak signal)
        host_dft_shift[0:wm_rows, cols - wm_cols : cols] += (
            0.5 * strength
        ) * np.fliplr(watermark_norm)

        # 4. Bottom-left corner (weak signal)
        host_dft_shift[rows - wm_rows : rows, 0:wm_cols] += (
            0.5 * strength
        ) * np.flipud(watermark_norm)

    elif region_type == "center":
        # Embed in center region only
        cx, cy = rows // 2, cols // 2
        center_region_size = int(min(rows, cols) * 0.3)  # 30% of image size
        _embed_center_region(
            host_dft_shift, watermark_norm, strength, cx, cy, center_region_size
        )

    elif region_type == "random":
        # Embed in random regions
        if random_seed is None:
            raise ValueError("random_seed must be provided when region_type='random'")
        _embed_random_regions(host_dft_shift, watermark_norm, strength, random_seed)

    else:
        raise ValueError(
            f"Invalid region_type: {region_type}. Must be one of: corners, center, random"
        )

    # Inverse transform
    host_idft_shift = np.fft.ifftshift(host_dft_shift)
    img_back = np.fft.ifft2(host_idft_shift)
    img_back = np.real(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_back)
    return output_path


def _extract_center_region(img_dft_shift, wm_shape, cx, cy, region_size):
    """Extract watermark from the center region of the frequency domain."""
    rows, cols = img_dft_shift.shape
    wm_rows, wm_cols = wm_shape

    # Calculate center region boundaries
    start_row = max(0, cx - region_size // 2)
    end_row = min(rows, cx + region_size // 2)
    start_col = max(0, cy - region_size // 2)
    end_col = min(cols, cy + region_size // 2)

    # Extract center region
    center_region = img_dft_shift[start_row:end_row, start_col:end_col]

    # Resize to watermark shape
    wm_resized = cv2.resize(
        np.abs(center_region), (wm_cols, wm_rows), interpolation=cv2.INTER_AREA
    )

    return wm_resized


def _extract_random_regions(img_dft_shift, wm_shape, seed, num_regions=5):
    """Extract watermark from random regions using specified seed."""
    rows, cols = img_dft_shift.shape
    wm_rows, wm_cols = wm_shape

    # Set random seed for reproducible extraction
    random.seed(seed)
    np.random.seed(seed)

    extracted_regions = []

    # Generate same random positions as embedding
    for i in range(num_regions):
        # Same random position as embedding
        start_row = random.randint(0, rows - wm_rows)
        start_col = random.randint(0, cols - wm_cols)

        # Extract region
        region = img_dft_shift[
            start_row : start_row + wm_rows, start_col : start_col + wm_cols
        ]
        extracted_regions.append(np.abs(region))

    # Average all extracted regions
    if extracted_regions:
        avg_region = np.mean(extracted_regions, axis=0)
        return avg_region
    else:
        return np.zeros((wm_rows, wm_cols))


def extract(
    watermarked_path: str | Path,
    *,
    strength: float = 10.0,
    scale: Optional[float] = 0.25,
    watermark_shape: Optional[Tuple[int, int]] = None,
    output_path: Optional[str | Path] = None,
    region_type: Literal["corners", "center", "random"] = "corners",
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """Extract watermark image from a watermarked picture with selectable regions.

    Parameters
    ----------
    watermarked_path : str | Path
        Path to watermarked image.
    strength : float, default 10.0
        Embedding strength used during embedding.
    scale : float, optional
        Watermark scale relative to host image.
    watermark_shape : tuple, optional
        Watermark shape (rows, cols).
    output_path : str | Path, optional
        Path to save extracted watermark.
    region_type : {"corners", "center", "random"}, default "corners"
        Type of regions used during embedding.
    random_seed : int, optional
        Random seed used during embedding (required when region_type="random").
    """
    img = _read_image(watermarked_path, gray=True)
    rows, cols = img.shape

    if scale is None and watermark_shape is None:
        raise ValueError(
            "Must provide either scale or watermark_shape to determine watermark size."
        )
    if scale is not None:
        wm_rows = int(rows * scale)
        wm_cols = int(cols * scale)
    else:
        wm_rows, wm_cols = watermark_shape  # type: ignore[misc]

    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)

    if region_type == "corners":
        # Extract from four corner regions
        regions = [
            img_dft_shift[0:wm_rows, 0:wm_cols],  # Top-left (main)
            img_dft_shift[rows - wm_rows : rows, cols - wm_cols : cols],  # Bottom-right
            img_dft_shift[0:wm_rows, cols - wm_cols : cols],  # Top-right
            img_dft_shift[rows - wm_rows : rows, 0:wm_cols],  # Bottom-left
        ]

        # Normalize all regions
        wm_candidates = [np.abs(r) / strength for r in regions]
        wm_candidates[1:] = [
            w / 0.5 for w in wm_candidates[1:]
        ]  # Bottom-right/Top-right/Bottom-left regions divide by 0.5

        # Normalize to 0-1
        normed = []
        for w in wm_candidates:
            w = w - w.min()
            w = w / w.max() if w.max() != 0 else w
            normed.append(w)

        # Fusion: main region has higher weight, corners are weaker
        fused = np.maximum(
            0.6 * normed[0], 0.133 * normed[1] + 0.133 * normed[2] + 0.133 * normed[3]
        )

    elif region_type == "center":
        # Extract from center region only
        cx, cy = rows // 2, cols // 2
        center_region_size = int(min(rows, cols) * 0.3)
        center = _extract_center_region(
            img_dft_shift, (wm_rows, wm_cols), cx, cy, center_region_size
        )

        # Normalize center region
        center = center / strength
        center = center - center.min()
        center = center / center.max() if center.max() != 0 else center
        fused = center

    elif region_type == "random":
        # Extract from random regions
        if random_seed is None:
            raise ValueError("random_seed must be provided when region_type='random'")
        fused = _extract_random_regions(img_dft_shift, (wm_rows, wm_cols), random_seed)

        # Normalize random regions
        fused = fused / strength
        fused = fused - fused.min()
        fused = fused / fused.max() if fused.max() != 0 else fused

    else:
        raise ValueError(
            f"Invalid region_type: {region_type}. Must be one of: corners, center, random"
        )

    fused = np.clip(fused, 0, 1)
    wm_uint8 = (fused * 255).astype(np.uint8)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), wm_uint8)
    return wm_uint8


def extract_text(
    watermarked_path: str | Path,
    *,
    strength: float = 10.0,
    scale: Optional[float] = 0.25,
    watermark_shape: Optional[Tuple[int, int]] = None,
    lang: str = "chi_sim+eng",
    tesseract_cmd: Optional[str] = None,
    region_type: Literal["corners", "center", "random"] = "corners",
    random_seed: Optional[int] = 42,
) -> str:
    """Extract text watermark from watermarked image.

    This function first calls :func:`extract` to get watermark image, then uses *Tesseract OCR* to recognize text.

    Note: requires `tesseract` executable installed, optionally specify full path via ``tesseract_cmd``.
    """
    # get watermark image (don't write to disk)
    wm_img = extract(
        watermarked_path,
        strength=strength,
        scale=scale,
        watermark_shape=watermark_shape,
        output_path=None,
        region_type=region_type,
        random_seed=random_seed,
    )

    # preprocessing: binarization + scaling
    img = wm_img.copy()
    # Otsu threshold
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Text usually needs black background and white text, invert if background is black
    if th.mean() < 128:
        th = cv2.bitwise_not(th)

    h, w = th.shape
    # Fixed 4x scaling for better OCR
    th = cv2.resize(th, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)

    pil_img = Image.fromarray(th)

    if tesseract_cmd is not None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        text = pytesseract.image_to_string(pil_img, lang=lang, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Could not find tesseract executable, please install Tesseract OCR or specify tesseract_cmd argument."
        ) from exc

    return text.strip()
