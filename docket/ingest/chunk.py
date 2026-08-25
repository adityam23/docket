"""Chunking with lineage. A chunk carries its origin (doc id, page, source path)
so every retrieved span resolves back to an exact citation (docs/decisions Q15).

Word-window chunking with overlap: dependency-free, deterministic, and good
enough for the lite any-PDF path. Page boundaries are respected so a chunk never
straddles two pages and its `page` is always exact for `cite_page`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str          # stable, unique within a corpus: f"{doc_id}#{n}"
    page: int              # 1-based source page
    text: str
    source: str = ""       # original file path

    def as_dict(self) -> dict:
        return asdict(self)


def _window(words: list[str], size: int, overlap: int):
    """Yield overlapping word windows. step = size - overlap (>=1)."""
    step = max(1, size - overlap)
    i = 0
    n = len(words)
    while i < n:
        yield words[i : i + size]
        if i + size >= n:
            break
        i += step


def chunk_pages(
    pages: list[dict],
    *,
    doc_id: str,
    source: str = "",
    words: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Chunk a document given as ``[{"page": int, "text": str}, ...]``.

    Chunks never cross a page boundary, so ``chunk.page`` is exact. Empty pages
    are skipped. ``chunk_id`` is a stable running index across the document.
    """
    out: list[Chunk] = []
    n = 0
    for pg in pages:
        page_no = int(pg.get("page", 1))
        toks = (pg.get("text") or "").split()
        if not toks:
            continue
        for w in _window(toks, words, overlap):
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{n}",
                    page=page_no,
                    text=" ".join(w),
                    source=source,
                )
            )
            n += 1
    return out


def chunk_text(
    text: str,
    *,
    doc_id: str,
    page: int = 1,
    source: str = "",
    words: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Chunk a single blob of text (single logical page)."""
    return chunk_pages(
        [{"page": page, "text": text}],
        doc_id=doc_id,
        source=source,
        words=words,
        overlap=overlap,
    )
