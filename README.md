# FDWM - Frequency Domain Watermarking

[![PyPI version](https://badge.fury.io/py/fdwm.svg)](https://badge.fury.io/py/fdwm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for embedding and extracting watermarks in images using frequency domain techniques. Supports both image and text watermarks with a command-line interface.

## Features

- **Frequency Domain Watermarking**: Uses FFT (Fast Fourier Transform) to embed watermarks in high-frequency regions
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

# Embed watermark
watermarked_img = embed(host_img, watermark_img, strength=0.1)

# Extract watermark
extracted_watermark = extract(watermarked_img, watermark_img.shape[:2])

# For text watermarks
text = "Hello World"
watermarked_img = embed(host_img, text, strength=0.1, is_text=True)
extracted_text = extract_text(watermarked_img)
```

### Command Line Interface

```bash
# Embed image watermark
fdwm embed host.jpg watermark.png -o watermarked.jpg

# Embed text watermark
fdwm embed host.jpg "Hello World" -o watermarked.jpg --text

# Extract image watermark
fdwm extract watermarked.jpg watermark.png -o extracted.png

# Extract text watermark
fdwm extract watermarked.jpg --text
```

## CLI Usage

### Embed Command

```bash
fdwm embed <host_image> <watermark> [options]

Options:
  -o, --output PATH     Output file path
  --text               Treat watermark as text
  --strength FLOAT     Embedding strength (default: 0.1)
  --batch              Process directory of images
```

### Extract Command

```bash
fdwm extract <watermarked_image> <watermark> [options]

Options:
  -o, --output PATH     Output file path
  --text               Extract text watermark
  --batch              Process directory of images
```

## Examples

### Batch Processing

```bash
# Embed watermark in all images in a directory
fdwm embed images/ watermark.png --batch -o watermarked/

# Extract watermark from all images
fdwm extract watermarked/ watermark.png --batch -o extracted/
```
g
### Text Watermarking

```bash
# Embed Chinese text
fdwm embed image.jpg "Hello World" --text -o watermarked.jpg

# Extract text
fdwm extract watermarked.jpg --text
```

## Requirements

- Python 3.8+
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