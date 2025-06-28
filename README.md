# FDWM - Frequency Domain Watermarking

[![PyPI version](https://badge.fury.io/py/fdwm.svg)](https://badge.fury.io/py/fdwm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for embedding and extracting watermarks in images using frequency domain techniques. Supports both image and text watermarks with a command-line interface.

## Features

- **Frequency Domain Watermarking**: Uses FFT (Fast Fourier Transform) to embed watermarks in high-frequency regions
- **Flexible Region Selection**: Choose from corners, center, or random regions for watermark embedding
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

# Embed watermark in corners (default)
watermarked_img = embed(host_img, watermark_img, strength=0.1, region_type="corners")

# Embed watermark in center region
watermarked_img = embed(host_img, watermark_img, strength=0.1, region_type="center")

# Embed watermark in random regions with seed
watermarked_img = embed(host_img, watermark_img, strength=0.1, region_type="random", random_seed=42)

# Extract watermark (use same region_type as embedding)
extracted_watermark = extract(watermarked_img, watermark_img.shape[:2], region_type="corners")

# For text watermarks
text = "Hello World"
watermarked_img = embed(host_img, text, strength=0.1, is_text=True, region_type="center")
extracted_text = extract_text(watermarked_img, region_type="center")
```

### Command Line Interface

```bash
# Embed image watermark in corners (default)
fdwm embed host.jpg --watermark-img watermark.png

# Embed image watermark in center region
fdwm embed host.jpg --watermark-img watermark.png --region-type center

# Embed image watermark in random regions
fdwm embed host.jpg --watermark-img watermark.png --region-type random --random-seed 42

# Embed text watermark
fdwm embed host.jpg --watermark-text "Hello World" --region-type center

# Extract image watermark (use same region_type as embedding)
fdwm extract watermarked.jpg --region-type center

# Extract text watermark
fdwm extract watermarked.jpg --text --region-type center
```

## Region Types

FDWM supports three different region types for watermark embedding:

### 1. Corners (Default)
Embeds watermark in the four corners of the image:
- **Top-left corner**: Main region with full strength
- **Bottom-right corner**: Flipped watermark with 50% strength
- **Top-right corner**: Horizontally flipped with 50% strength
- **Bottom-left corner**: Vertically flipped with 50% strength

This provides good robustness against cropping attacks.

### 2. Center
Embeds watermark in the center region of the image (30% of image size).
- Provides good resistance to edge cropping
- Maintains watermark integrity in central areas

### 3. Random
Embeds watermark in random regions using a specified seed:
- Uses 5 random positions by default
- Applies random transformations (flips)
- Requires `random_seed` parameter for reproducible results
- Provides high security through obfuscation

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
  --region-type TYPE      Region type: corners, center, random (default: corners)
  --random-seed INT       Random seed for reproducible random embedding
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
  --region-type TYPE      Region type used during embedding: corners, center, random (default: corners)
  --random-seed INT       Random seed used during embedding
```

## Examples

### Different Region Types

```bash
# Embed in corners (most robust)
fdwm embed image.jpg --watermark-text "Secret" --region-type corners

# Embed in center (resistant to edge cropping)
fdwm embed image.jpg --watermark-text "Secret" --region-type center

# Embed in random regions (high security)
fdwm embed image.jpg --watermark-text "Secret" --region-type random --random-seed 12345

# Extract using same region type
fdwm extract watermarked.jpg --text --region-type center
```

### Batch Processing

```bash
# Embed watermark in all images in a directory
fdwm embed images/ --watermark-img watermark.png --region-type center

# Extract watermark from all images
fdwm extract watermarked/ --region-type center --output extracted/
```

### Text Watermarking

```bash
# Embed Chinese text in center region
fdwm embed image.jpg --watermark-text "Hello World" --region-type center

# Extract text
fdwm extract watermarked.jpg --text --region-type center
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
- Uses OpenCV for image processing
- Tesseract OCR for text extraction