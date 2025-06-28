# FDWM - Frequency Domain Watermarking

[![PyPI version](https://badge.fury.io/py/fdwm.svg)](https://badge.fury.io/py/fdwm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for embedding and extracting watermarks in images using frequency domain techniques. Supports both image and text watermarks with a command-line interface.

## Features

- **Frequency Domain Watermarking**: Uses FFT (Fast Fourier Transform) to embed watermarks in high-frequency regions
- **Grid-based Embedding**: Embeds watermark in multiple grid cells for robust extraction
- **Image Watermarks**: Embed and extract image-based watermarks
- **Text Watermarks**: Embed and extract text watermarks with OCR support
- **Command Line Interface**: Easy-to-use CLI for batch processing
- **Multiple Languages**: Support for various text languages
- **Robust Extraction**: High correlation coefficients for reliable watermark detection

## Installation

### From PyPI (Recommended)

```bash
pip install fdwm
```

### From Source

```bash
git clone https://github.com/Liam0205/fdwm.git
cd fdwm
pip install -e .
```

## Quick Start

### Python API

```python
import cv2
from fdwm import embed, extract, extract_text

# Load host image and watermark
host_img = cv2.imread('host.jpg')
watermark_img = cv2.imread('watermark.png')

# Embed watermark using grid strategy (3x3 grid by default)
out_path, metrics = embed(host_img, watermark_img, strength=0.1)
print("Watermarked image saved to:", out_path)
print("Metrics:", metrics)

# Extract watermark
extracted_watermark = extract(out_path, watermark_img.shape[:2])

# For text watermarks
text = "Hello World"
out_path, metrics = embed(host_img, None, 'watermarked.png', watermark_text=text, strength=0.1)
extracted_text = extract_text(out_path)
```

**Note:**
- `embed` returns a tuple `(output_path, metrics)`.
- `metrics` is a dict with keys: `'mean_pixel_diff'`, `'max_pixel_diff'`, `'p90_pixel_diff'`, `'psnr'`.

### Command Line Interface

```bash
# Embed image watermark (3x3 grid by default)
fdwm embed host.jpg --watermark-img watermark.png

# Embed with custom grid size
fdwm embed host.jpg --watermark-img watermark.png --grid-m 4 --grid-n 4

# Embed text watermark
fdwm embed host.jpg --watermark-text "Hello World"

# Extract image watermark
fdwm extract watermarked.jpg

# Extract text watermark
fdwm extract watermarked.jpg --text
```

## Embedding Method

FDWM uses a grid-based embedding strategy:

1. **Grid Division**: The image is divided into `m × n` grid cells (default: 3×3)
2. **Cell Processing**: Each grid cell is processed independently in the frequency domain
3. **Corner Embedding**: Within each cell, the watermark is embedded in the top-left corner
4. **Fusion**: During extraction, all grid cells are processed and averaged for robust recovery

This approach provides excellent robustness against cropping, rotation, and other geometric attacks.

## CLI Usage

### Embed Command

```bash
fdwm embed <host_image> [options]

Options:
  --watermark-img PATH    Watermark image path
  --watermark-text TEXT   Watermark text
  --strength FLOAT        Embedding strength (default: 30000.0)
  --scale FLOAT           Watermark scale relative to host (default: 0.25)
  --font PATH             Font file path for text watermark
  --font-size INT         Font size for text watermark
  --grid-m INT            Number of vertical grid divisions (default: 3)
  --grid-n INT            Number of horizontal grid divisions (default: 3)
  --debug                 Print detailed metrics for each processed image
```

### Extract Command

```bash
fdwm extract <watermarked_image> [options]

Options:
  --strength FLOAT        Embedding strength used during embedding (default: 30000.0)
  --scale FLOAT           Watermark scale relative to host (default: 0.25)
  --output PATH           Directory to save extracted watermark images/text
  --text                  Perform OCR and output text instead of image
  --save-text             Save recognized text to .txt files
  --grid-m INT            Number of vertical grid divisions used during embedding (default: 3)
  --grid-n INT            Number of horizontal grid divisions used during embedding (default: 3)
```

## Examples

### Basic Usage

```bash
# Embed watermark with default 3x3 grid
fdwm embed image.jpg --watermark-text "Secret"

# Embed with 4x4 grid for higher robustness
fdwm embed image.jpg --watermark-text "Secret" --grid-m 4 --grid-n 4

# Extract text
fdwm extract watermarked.jpg --text
```

### Batch Processing

```bash
# Embed watermark in all images in a directory
fdwm embed images/ --watermark-img watermark.png

# Extract watermark from all images
fdwm extract watermarked/ --output extracted/
```

### Text Watermarking

```bash
# Embed Chinese text
fdwm embed image.jpg --watermark-text "Hello World"

# Extract text
fdwm extract watermarked.jpg --text
```

## Requirements

- Python 3.10+
- numpy
- opencv-python
- Pillow
- pytesseract

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Based on frequency domain watermarking techniques