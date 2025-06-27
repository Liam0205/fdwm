import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pytesseract
import tempfile


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
                    font = ImageFont.truetype(fallback_font_path, font_size) if fallback_font_path else ImageFont.load_default()
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
            draw.multiline_text((x, y), wrapped, fill=255, font=font, spacing=4, align="center")
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

    Use 2-D FFT, add watermark in high-frequency corner of host image.

    Parameters
    ----------
    host_path : str | pathlib.Path
        Host image path.
    watermark_path : str | pathlib.Path | None
        Watermark image path (grayscale recommended). Ignored if *watermark_text* given.
    output_path : str | pathlib.Path
        Destination path to write the watermarked image (will overwrite if identical).
    strength : float, default ``10.0``
        Embedding strength. Larger → watermark easier to extract but less invisible.
    scale : float, default ``0.25``
        Watermark size as a ratio of host image (0–1).
    font_path : str | None, optional
        Font file path. If ``None``, use default font.
    font_size : int | None, optional
        Font size. If ``None``, auto-estimate.

    Returns
    -------
    pathlib.Path
        The saved image path.
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
        # Resize watermark to target size
        watermark = cv2.resize(
            watermark_src, (wm_cols, wm_rows), interpolation=cv2.INTER_AREA
        )

    # normalize watermark to [0, 1] range for strength adjustment
    watermark_norm = watermark.astype(np.float32) / 255.0

    # compute host image spectrum
    host_dft = np.fft.fft2(host)
    host_dft_shift = np.fft.fftshift(host_dft)

    # --- embed position changed to spectrum corner (highest frequency), reduce visual impact ---
    r_start = 0
    c_start = 0
    r_end = wm_rows
    c_end = wm_cols

    # symmetric position
    r_start_sym = rows - wm_rows
    c_start_sym = cols - wm_cols
    r_end_sym = rows
    c_end_sym = cols

    # directly add watermark at high frequency (amplitude addition)
    host_dft_shift[r_start:r_end, c_start:c_end] += strength * watermark_norm
    host_dft_shift[r_start_sym:r_end_sym, c_start_sym:c_end_sym] += (
        strength * np.flipud(np.fliplr(watermark_norm))
    )

    # inverse FFT back to spatial domain
    host_idft_shift = np.fft.ifftshift(host_dft_shift)
    img_back = np.fft.ifft2(host_idft_shift)
    img_back = np.real(img_back)

    # limit pixel values to [0, 255]
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    # save result
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

    If *scale* provided, watermark size is inferred; otherwise specify *watermark_shape*.

    Parameters
    ----------
    watermarked_path : str | pathlib.Path
        Path to watermarked image.
    strength : float, optional, default ``10.0``
        Same ``strength`` value as used during embedding.
    scale : float | None, optional
        Same ``scale`` as used during embedding. If ``None``, must provide ``watermark_shape``.
    watermark_shape : tuple[int, int] | None, optional
        Watermark dimensions (rows, cols). Ignored if ``scale`` is provided.
    output_path : str | pathlib.Path | None, optional
        Path to save extracted watermark. If ``None``, don't save.

    Returns
    -------
    numpy.ndarray
        Extracted watermark image (uint8).
    """
    img = _read_image(watermarked_path, gray=True)
    rows, cols = img.shape

    if scale is None and watermark_shape is None:
        raise ValueError("Must provide either scale or watermark_shape to determine watermark size.")

    if scale is not None:
        wm_rows = int(rows * scale)
        wm_cols = int(cols * scale)
    else:
        wm_rows, wm_cols = watermark_shape  # type: ignore[misc]

    # compute spectrum
    img_dft = np.fft.fft2(img)
    img_dft_shift = np.fft.fftshift(img_dft)

    # same high-frequency corner as embedding
    r_start = 0
    c_start = 0
    r_end = wm_rows
    c_end = wm_cols

    wm_freq = img_dft_shift[r_start:r_end, c_start:c_end]

    # directly restore using addition model
    wm_mag = np.abs(wm_freq) / strength

    # normalize to 0–255 for visualization
    wm_norm = wm_mag - wm_mag.min()
    if wm_norm.max() != 0:
        wm_norm = wm_norm / wm_norm.max()
    wm_uint8 = (wm_norm * 255).astype(np.uint8)

    # save result
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
    th = cv2.resize(th, (w*4, h*4), interpolation=cv2.INTER_NEAREST)

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