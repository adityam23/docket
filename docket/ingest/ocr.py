"""PDF → text with page lineage.

Two tiers, chosen per page so the base install stays light:

1. **Born-digital PDFs** — extracted with ``pypdf`` (lazy import; comes with the
   ``ingest`` extra). No model, no GPU, works offline. This is the common
   any-PDF case (Q5) and needs no download.
2. **Scanned / image-only pages** — fall back to mettle's ``/v1/ocr`` (see
   ``_ocr_model.py``). The rasterizer-only import (pymupdf) is lazy and loaded
   *only* when a page yields no extractable text, so we never pull it in for a
   text PDF.
"""

from __future__ import annotations


def pdf_to_pages(path: str, *, on_missing_text: str = "ocr") -> list[dict]:
    """Return ``[{"page": 1-based int, "text": str}, ...]`` for a PDF.

    Uses pypdf for text. A page with no extractable text is handled per
    ``on_missing_text``:

    * ``"ocr"`` (default) — route it through the OCR fallback, raising a clear,
      actionable error if the engine's OCR call fails. This is the strict
      production behaviour: a scanned/image filing must not be ingested with
      pages silently missing.
    * ``"skip"`` — emit empty text for that page and keep the rest. For callers
      that only need the born-digital text (e.g. the generator-free retrieval
      sweep, whose gold evidence lives on text pages), one image page — a chart,
      a signature block — must not drop the whole filing.
    """
    if on_missing_text not in ("ocr", "skip"):
        raise ValueError(f"on_missing_text must be 'ocr' or 'skip', got {on_missing_text!r}")
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - exercised via the ingest extra
        raise RuntimeError(
            "PDF text extraction needs pypdf — run `uv sync --extra ingest`"
        ) from e

    reader = PdfReader(path)
    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text and on_missing_text == "ocr":
            text = _ocr_page_fallback(path, i)
        pages.append({"page": i, "text": text})
    return pages


def _ocr_page_fallback(path: str, page: int) -> str:
    """Scanned page → mettle's ``/v1/ocr``. ``_ocr_model`` always ships (its
    only lazy import, pymupdf, is inside ``ocr_page`` itself) so this calls
    straight through — no import guard to fall back on."""
    import docket.ingest._ocr_model as _m

    return _m.ocr_page(path, page)  # pragma: no cover - requires a live engine
