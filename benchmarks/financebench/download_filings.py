#!/usr/bin/env python
"""Download the source filing PDFs referenced by the FinanceBench open subset.

FinanceBench full-RAG mode needs one ``<doc_name>.pdf`` per question's filing.
This fetches each unique ``doc_link`` from the merged dataset into ``--out-dir``,
with a browser UA, redirect following, retries, and a %PDF-header validity check,
then reports coverage (these are public SEC filings / investor-relations PDFs).

    uv run python benchmarks/financebench/download_filings.py \
        --out-dir benchmarks/financebench/filings

Idempotent: an already-valid PDF is skipped, so re-running only retries misses.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CACHE = Path(__file__).with_name("data") / "financebench_merged.jsonl"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
# Canonical, reliable source: the FinanceBench repo hosts every filing PDF as
# pdfs/<doc_name>.pdf. Tried first; the per-record doc_link (investor-relations
# hosts that time out / DNS-fail / serve HTML landing pages) is the fallback.
GH_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{name}.pdf"


def _unique_docs(records: list[dict]) -> dict[str, str]:
    """doc_name -> doc_link (first link wins; they're identical per filing)."""
    out: dict[str, str] = {}
    for r in records:
        name, link = r.get("doc_name"), r.get("doc_link")
        if name and link and name not in out:
            out[name] = link
    return out


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF")
    except OSError:
        return False


def _fetch(name: str, link: str, out_dir: Path, tries: int = 3) -> tuple[str, bool, str]:
    dest = out_dir / f"{name}.pdf"
    if dest.is_file() and _is_pdf(dest) and dest.stat().st_size > 4096:
        return name, True, "cached"
    # Try the canonical GitHub-raw copy first, then the record's doc_link.
    urls = [GH_RAW.format(name=name), link]
    last = ""
    for url in urls:
        for attempt in range(tries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                    data = resp.read()
                if not data.startswith(b"%PDF"):
                    last = f"not a PDF (got {data[:16]!r})"
                    break  # wrong content type — try the next source, not more retries
                dest.write_bytes(data)
                src = "github" if url == urls[0] else "doc_link"
                return name, True, f"{len(data) // 1024} KB ({src})"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
    return name, False, last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(Path(__file__).with_name("filings")))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.cache) if l.strip()]
    docs = _unique_docs(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(docs)} unique filings -> {out_dir}")

    ok: list[str] = []
    fail: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch, n, l, out_dir): n for n, l in docs.items()}
        for fut in as_completed(futs):
            name, good, note = fut.result()
            print(f"  [{'ok ' if good else 'FAIL'}] {name:<40} {note}")
            (ok if good else fail).append(name if good else (name, note))

    print(f"\n{len(ok)}/{len(docs)} downloaded. {len(fail)} failed.")
    if fail:
        print("failed filings (need a manual/alternate source):")
        for name, note in fail:
            print(f"  {name}: {note}")


if __name__ == "__main__":
    main()
