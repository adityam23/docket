"""PDF → text with page lineage.

Two tiers, chosen per page so the base install stays light:

1. **Born-digital PDFs** — extracted with ``pypdf`` (lazy import; comes with the
   ``ingest`` extra). No model, no GPU, works offline. This is the common
   any-PDF case (Q5) and needs no download.
2. **Scanned / image-only pages** — fall back to a real OCR model (Baidu
   Unlimited-OCR, or Surya/dots.ocr to fit 6 GB). Loaded lazily *only* when a
   page yields no extractable text, so we never pull multi-GB weights for a
   text PDF. Not installed on this disk-constrained host → a clear, actionable
   error names the missing piece rather than failing opaquely.
"""

from __future__ import annotations


def pdf_to_pages(path: str, *, on_missing_text: str = "ocr") -> list[dict]:
    """Return ``[{"page": 1-based int, "text": str}, ...]`` for a PDF.

    Uses pypdf for text. A page with no extractable text is handled per
    ``on_missing_text``:

    * ``"ocr"`` (default) — route it through the OCR fallback, raising a clear,
      actionable error if no OCR model is installed. This is the strict
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


def ocr_pdf(path: str) -> str:
    """Convenience: whole-document markdown-ish text (pages joined by rule)."""
    return "\n\n---\n\n".join(p["text"] for p in pdf_to_pages(path))


def _ocr_page_fallback(path: str, page: int) -> str:
    """Scanned page → OCR model. Lazy so text PDFs never trigger a model load."""
    try:
        # Drop-in seam: a dedicated OCR service/model (Unlimited-OCR / Surya).
        # Kept import-guarded; no OCR model served here yet.
        import docket.ingest._ocr_model as _m  # type: ignore
    except ImportError:
        raise RuntimeError(
            f"page {page} of {path} has no embedded text (scanned/image PDF) and "
            "no OCR model is installed. Serve an OCR model at /v1/ocr "
            "or pre-OCR the file."
        ) from None
    return _m.ocr_page(path, page)  # pragma: no cover - requires OCR weights
