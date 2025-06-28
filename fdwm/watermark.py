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
) -> Path:
    """Embed watermark in frequency domain.

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

    # Inverse transform
    host_idft_shift = np.fft.ifftshift(host_dft_shift)
    img_back = np.fft.ifft2(host_idft_shift)
    img_back = np.real(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_back)
    return output_path


def extract(
    watermarked_path: str | Path,
    *,
    strength: float = 10.0,
    scale: Optional[float] = 0.25,
    watermark_shape: Optional[Tuple[int, int]] = None,
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Extract watermark image from a watermarked picture.

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

    # Extract from four corner regions
    regions = [
        img_dft_shift[0:wm_rows, 0:wm_cols],  # Top-left (main)
        img_dft_shift[rows - wm_rows : rows, cols - wm_cols : cols],  # Bottom-right
        img_dft_shift[0:wm_rows, cols - wm_cols : cols],  # Top-right
        img_dft_shift[rows - wm_rows : rows, 0:wm_cols],  # Bottom-left
    ]

    # Extract magnitude from each region
    wm_candidates = []
    for i, r in enumerate(regions):
        magnitude = np.abs(r)
        # Apply strength normalization
        if i == 0:  # Main region (top-left)
            normalized = magnitude / strength
        else:  # Corner regions
            normalized = magnitude / (0.5 * strength)
        # Apply inverse transformations for corner regions
        if i == 1:  # Bottom-right: flip both axes
            normalized = np.flipud(np.fliplr(normalized))
        elif i == 2:  # Top-right: flip horizontally
            normalized = np.fliplr(normalized)
        elif i == 3:  # Bottom-left: flip vertically
            normalized = np.flipud(normalized)
        wm_candidates.append(normalized)

    # Average all regions
    fused = np.mean(wm_candidates, axis=0)

    # Normalize
    fused = np.clip(fused, 0, None)
    if fused.max() > 0:
        fused = fused / fused.max()

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
