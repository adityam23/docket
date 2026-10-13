"""`dk` command-line entry point (secondary surface; the primary UX is the web
app)."""

from __future__ import annotations

import typer

from .config import load_settings
from .providers.router import get_provider

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,  # clean one-line errors, no rich traceback
    help="Docket — a self-hosted research assistant for SEC filings: answers cited to the exact filing page, from a local model.",
)


@app.command()
def health() -> None:
    """Check the engine (mettle) and the configured model surface."""
    from .service import health as health_service

    info = health_service()
    status = "OK" if info.get("ok") else "FAIL"
    typer.echo(f"[{info['provider']}] {status} — {info}")


@app.command()
def chat(prompt: str) -> None:
    """One-shot chat against the backend (smoke test)."""
    s = load_settings()
    res = get_provider(s).chat([{"role": "user", "content": prompt}], max_tokens=256)
    typer.echo(res.text)


@app.command()
def embed(text: str) -> None:
    """Embed a string through the engine (smoke test)."""
    s = load_settings()
    vec = get_provider(s).embed([text])[0]
    typer.echo(f"dim={len(vec)} head={[round(x, 4) for x in vec[:5]]}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8760, reload: bool = True) -> None:
    """Run the local dashboard + observability API.

    Serves the built SvelteKit SPA at ``/`` (build it once with
    ``cd docket/web/frontend && npm install && npm run build``) and the
    JSON API at ``/api/*``. Falls back to a "not built" page until the build
    exists.
    """
    import uvicorn

    uvicorn.run("docket.web.app:app", host=host, port=port, reload=reload)


@app.command()
def ingest(folder: str) -> None:
    """Ingest every PDF under FOLDER: OCR→chunk→post to the engine collection.

    A document whose POST fails (e.g. the engine's model slot dropped
    mid-batch) is skipped with its reason printed — the rest of the folder
    still ingests; rerun the command to retry just the failed document(s)
    (contract D3)."""
    from .ingest.pipeline import ingest_folder

    s = load_settings()
    failed: list[tuple[str, str]] = []

    def on_event(doc_id: str, stage: str, info: dict) -> None:
        if stage == "failed":
            failed.append((doc_id, str(info.get("reason", ""))))

    corpus = ingest_folder(folder, settings=s, on_event=on_event)
    typer.echo(f"indexed {len(corpus)} chunks → {s.index_dir} "
               f"(lineage rows → {s.index_dir}; chunks live engine-side)")
    if failed:
        typer.echo(f"\n{len(failed)} document(s) failed and were skipped (retryable):", err=True)
        for doc_id, reason in failed:
            typer.echo(f"  - {doc_id}: {reason}", err=True)


@app.command()
def docs() -> None:
    """List indexed documents and how many more this device can hold."""
    from .ingest.index import Corpus
    from .web.observability import capacity_stats, corpus_stats

    s = load_settings()
    corpus = Corpus.load(s.index_dir)  # load once; hand it to both read models
    stats = corpus_stats(s, corpus=corpus)
    for d in stats["documents"]:
        typer.echo(f"  {d['doc_id']:<48} {d['pages']:>4}p {d['chunks']:>4}c  [{d['stage']}]")
    t = stats["totals"]
    typer.echo(f"\n{t['documents']} documents · {t['chunks']} chunks")

    cap = capacity_stats(s, corpus=corpus)
    free_gb = cap["device"]["disk_free_bytes"] / 1024**3
    typer.echo(
        f"~{cap['remaining_documents_est']:,} more documents fit "
        f"({free_gb:.1f} GB free, ~{cap['bytes_per_document'] / 1000:.0f} kB/doc"
        f"{'' if cap['bytes_per_document_measured'] else ', estimated'})."
    )


@app.command()
def remove(doc_id: str) -> None:
    """Remove a document from the local index (see `dk docs` for ids)."""
    from .service import remove_document

    s = load_settings()
    try:
        removed = remove_document(doc_id, settings=s)
    except RuntimeError as e:
        # The engine is out of sync with our lineage -- a clean one-line error,
        # not an uncaught traceback (lineage is left in place for a retry).
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    if removed:
        typer.echo(f"removed {doc_id} ({removed} chunks)")
    else:
        typer.echo(f"no such document: {doc_id}")
        raise typer.Exit(code=1)


@app.command()
def samples() -> None:
    """Load the bundled synthetic sample documents (opt-in demo corpus)."""
    from .ingest.samples import load_samples

    s = load_settings()
    corpus = load_samples(settings=s)
    typer.echo(f"loaded samples → {len(corpus)} chunks total in {s.index_dir}")


@app.command()
def ask(question: str) -> None:
    """Answer a question over the ingested corpus (grounded, cited, scored)."""
    from .report.render import reliability_banner
    from .service import ask as ask_service

    res = ask_service(question)
    typer.echo(reliability_banner(res.reliability))
    typer.echo("")
    typer.echo(res.answer)
    if res.citations:
        typer.echo("\nSources:")
        for c in res.citations:
            typer.echo(f"  - {c.doc_id} p.{c.page}")


# TODO(phase-1): `report`, `eval`, `bench`.


def main() -> None:
    app()


if __name__ == "__main__":
    main()
