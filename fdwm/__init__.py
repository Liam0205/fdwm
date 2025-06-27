"""Frequency-domain watermarking library and CLI."""

__version__ = "0.1.0"
__author__ = "Liam Huang"
__email__ = "PyPI@liam.page"

from .watermark import embed, extract, extract_text  # noqa: F401

__all__ = [
    "embed",
    "extract",
    "extract_text",
]