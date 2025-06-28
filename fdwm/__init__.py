"""Frequency-domain watermarking library."""

__version__ = "0.2.2"
__author__ = "Liam Huang"
__email__ = "PyPI@liam.page"

from .watermark import embed, extract, extract_text  # noqa: F401

__all__ = [
    "embed",
    "extract",
    "extract_text",
]
