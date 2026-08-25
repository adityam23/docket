"""Scanned-page OCR through a `/v1/ocr` endpoint (Baidu Unlimited-OCR served by
a llama.cpp-based engine). The consumer never touches PDFs on
the server side — this module rasterizes each page locally and posts the
encoded image, per the OCR contract (image in, text out).

Lazy imports keep the base install light: pymupdf is only needed when a
page actually falls back to OCR, and the module is only imported then
(see ocr.py `_ocr_page_fallback`).
"""

from __future__ import annotations

import base64

import httpx

from ..config import load_settings


def ocr_page(path: str, page: int) -> str:
    """OCR one 1-based page of a PDF via the configured backend.

    Rasterizes at ~200 DPI (the OCR model's sweet spot without bloating the
    request), encodes PNG, and POSTs to ``{DK_BACKEND_URL}/ocr`` (the URL
    already ends in ``/v1``).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "scanned-page OCR needs PyMuPDF — install it (`uv add pymupdf`); "
            "it is used only for pages without embedded text"
        ) from e

    settings = load_settings()
    endpoint = f"{settings.backend_url.rstrip('/')}/ocr"

    doc = fitz.open(path)
    try:
        pix = doc[page - 1].get_pixmap(dpi=200)
        png = pix.tobytes("png")
    finally:
        doc.close()
    encoded = base64.b64encode(png).decode("ascii")

    try:
        r = httpx.post(endpoint, json={"image": encoded}, timeout=120.0)
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"OCR backend {endpoint} failed: HTTP {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"OCR backend {endpoint} unreachable: {e}") from e

    if not text:
        raise RuntimeError(f"OCR backend returned no text for page {page} of {path}")
    return text
