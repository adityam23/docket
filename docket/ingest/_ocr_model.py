"""Scanned-page OCR through mettle's `/v1/ocr` endpoint. The consumer never
touches PDFs on the server side — this module rasterizes each page locally
and hands the encoded image to the shared OpenAI-/v1 client (bearer auth,
configured timeout — see ``providers/openai_compat.py``), per the OCR
contract (image in, text out).

Lazy imports keep the base install light: pymupdf is only needed when a
page actually falls back to OCR, and the module is only imported then
(see ocr.py `_ocr_page_fallback`).
"""

from __future__ import annotations

import base64

from ..config import load_settings
from ..providers.router import get_provider


def ocr_page(path: str, page: int) -> str:
    """OCR one 1-based page of a PDF via mettle's ``/v1/ocr``.

    Rasterizes at ~200 DPI (the OCR model's sweet spot without bloating the
    request), encodes PNG, and calls the shared provider's ``ocr()`` — the
    same client every other /v1 call uses, so this gets the bearer header and
    the configured request timeout for free instead of a bare, unauthenticated
    ``httpx.post`` with a hard-coded timeout.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "scanned-page OCR needs PyMuPDF — install it (`uv add pymupdf`); "
            "it is used only for pages without embedded text"
        ) from e

    settings = load_settings()
    doc = fitz.open(path)
    try:
        pix = doc[page - 1].get_pixmap(dpi=200)
        png = pix.tobytes("png")
    finally:
        doc.close()
    encoded = base64.b64encode(png).decode("ascii")

    text = get_provider(settings).ocr(encoded).strip()
    if not text:
        raise RuntimeError(f"OCR backend returned no text for page {page} of {path}")
    return text
