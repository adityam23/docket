"""`dk` command-line entry point (secondary surface; the primary UX is the web
app)."""

from __future__ import annotations

import typer

from .config import load_settings
from .providers.router import get_provider

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,  # clean one-line errors, no rich traceback
    help="docket — self-hostable agentic document engine",
)


@app.command()
def health() -> None:
    """Check the configured backend (llama-server / Ollama / an API provider)."""
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
    """Embed a string (needs DK_EMBED_URL configured)."""
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
    """Ingest every PDF under FOLDER into the local index (OCR→chunk→embed→index)."""
    from .ingest.pipeline import ingest_folder

    s = load_settings()
    corpus = ingest_folder(folder, settings=s)
    typer.echo(f"indexed {len(corpus)} chunks → {s.index_dir} "
               f"(dense={'on' if corpus.has_vectors else 'off (sparse-only)'})")


@app.command()
def docs() -> None:
    """List indexed documents and how many more this device can hold."""
    from .web.observability import capacity_stats, corpus_stats

    s = load_settings()
    stats = corpus_stats(s)
    for d in stats["documents"]:
        typer.echo(f"  {d['doc_id']:<48} {d['pages']:>4}p {d['chunks']:>4}c  [{d['stage']}]")
    t = stats["totals"]
    typer.echo(f"\n{t['documents']} documents · {t['chunks']} chunks")

    cap = capacity_stats(s, corpus=None)
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
    removed = remove_document(doc_id, settings=s)
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
