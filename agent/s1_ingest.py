"""S1 — Ingestion. PyMuPDF text extraction, per-page scan detection, content-hash cache.

docs/FINDINGS-2026-08-06.md §1: 199/200 open-set PDFs have a text layer; parsing all 200
takes 15.8s single-threaded — not the bottleneck. No dedicated OCR stack (tesseract/marker/
docling) is built; a page is routed to the vision fallback purely by a per-page character
count threshold (a scan detector, not a parser choice).
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from agent.normalize import sanitize
from agent.schemas import IngestedDocument, PageText

_PLAINTEXT_EXTS = {".txt", ".md"}
_SUPPORTED_EXTS = {".pdf"} | _PLAINTEXT_EXTS
# Seen but not parsed without extra deps — surfaced in failures for triage.
_UNSUPPORTED_NOTED = {".docx", ".xlsx", ".doc", ".xls"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Sparse pages that still embed a raster table/figure (Fable V2: scans-inside-text PDFs).
_SPARSE_IMAGE_CHAR_CEILING = 200


def _page_has_raster(page: fitz.Page) -> bool:
    """True when the page embeds at least one image (table scan, stamp, figure)."""
    try:
        return bool(page.get_images(full=True))
    except Exception:  # noqa: BLE001 — treat as no raster
        return False


def _parse_pdf(path: Path, vision_char_threshold: int) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text()
            normalized = sanitize(raw)
            char_count = len(normalized.strip())
            # Full-scan pages OR sparse text + embedded raster (add-back / pledge tables).
            is_scan = char_count < vision_char_threshold or (
                char_count < _SPARSE_IMAGE_CHAR_CEILING and _page_has_raster(page)
            )
            pages.append(
                PageText(
                    page_number=i + 1,
                    raw_text=raw,
                    normalized_text=normalized,
                    char_count=char_count,
                    is_scan=is_scan,
                )
            )
    return pages


def _write_parsed_cache(cache_file: Path, doc: IngestedDocument) -> None:
    """Atomic write: tmp → os.replace (safe under ProcessPool concurrent writers)."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp.write_text(doc.model_dump_json(), encoding="utf-8")
    os.replace(tmp, cache_file)


def ingest_pdf(
    path: Path,
    cache_dir: Path | None = None,
    vision_char_threshold: int = 50,
    use_cache: bool = True,
) -> IngestedDocument:
    """Parses a single PDF, cached by SHA-256 of the file bytes."""
    digest = sha256_file(path)
    cache_file = (cache_dir / "parsed" / f"{digest}.json") if cache_dir else None

    if use_cache and cache_file and cache_file.exists():
        return IngestedDocument.model_validate_json(cache_file.read_text(encoding="utf-8"))

    pages = _parse_pdf(path, vision_char_threshold)
    doc = IngestedDocument(doc_id=path.stem, path=str(path), sha256=digest, pages=pages)

    if use_cache and cache_file:
        _write_parsed_cache(cache_file, doc)

    return doc


def ingest_plaintext(path: Path) -> IngestedDocument:
    """Single-page text/markdown document (noise notes, organizer README, etc.)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    normalized = sanitize(raw)
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    page = PageText(
        page_number=1,
        raw_text=raw,
        normalized_text=normalized,
        char_count=len(normalized.strip()),
        is_scan=False,
    )
    return IngestedDocument(doc_id=path.stem, path=str(path), sha256=digest, pages=[page])


def _ingest_pdf_worker(
    args: tuple[str, str | None, int, bool],
) -> tuple[IngestedDocument | None, str | None]:
    """Picklable ProcessPool worker. Returns (doc, failure_message)."""
    path_s, cache_s, threshold, use_cache = args
    path = Path(path_s)
    try:
        doc = ingest_pdf(
            path,
            cache_dir=Path(cache_s) if cache_s else None,
            vision_char_threshold=threshold,
            use_cache=use_cache,
        )
        return doc, None
    except Exception as exc:  # noqa: BLE001 — one bad PDF must not kill the run
        return None, f"{path.name}: {type(exc).__name__}: {exc}"


def ingest_directory(
    documents_dir: Path,
    cache_dir: Path | None = None,
    vision_char_threshold: int = 50,
    use_cache: bool = True,
    max_workers: int | None = None,
) -> tuple[list[IngestedDocument], list[str]]:
    """Ingest supported files in `documents_dir`. Returns (docs, failures).

    Parallel PDF parse via ProcessPoolExecutor; plaintext ingested in-process.
    Order of successful docs is sorted by path name.
    """
    docs: list[IngestedDocument] = []
    failures: list[str] = []
    if not documents_dir.is_dir():
        return [], [f"{documents_dir}: not a directory"]

    files = sorted(p for p in documents_dir.iterdir() if p.is_file())
    pdf_paths: list[Path] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in _UNSUPPORTED_NOTED:
            failures.append(f"{path.name}: формат не поддержан ({suffix})")
            continue
        if suffix not in _SUPPORTED_EXTS:
            continue
        if suffix in _PLAINTEXT_EXTS:
            try:
                docs.append(ingest_plaintext(path))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        pdf_paths.append(path)

    if pdf_paths:
        workers = max_workers if max_workers is not None else min(os.cpu_count() or 4, 8)
        workers = max(1, min(workers, len(pdf_paths)))
        cache_s = str(cache_dir) if cache_dir else None
        work = [(str(p), cache_s, vision_char_threshold, use_cache) for p in pdf_paths]

        if workers == 1:
            results = [_ingest_pdf_worker(w) for w in work]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_ingest_pdf_worker, work))

        # Preserve path order among successes.
        by_path = {str(p): (doc, err) for p, (doc, err) in zip(pdf_paths, results)}
        for p in pdf_paths:
            doc, err = by_path[str(p)]
            if err:
                failures.append(err)
            elif doc is not None:
                docs.append(doc)

    docs.sort(key=lambda d: d.path)
    return docs, failures


def _main() -> int:
    import argparse

    from agent.dataset import discover_dataset

    p = argparse.ArgumentParser(description="S1 ingestion smoke run: parse all PDFs, print coverage stats")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=Path(".cache"))
    args = p.parse_args()

    paths = discover_dataset(args.dataset)
    docs, failures = ingest_directory(paths.documents, cache_dir=args.cache)
    n_scan = sum(1 for d in docs if d.is_scan)
    n_pages = sum(len(d.pages) for d in docs)
    print(f"parsed {len(docs)} docs, {n_pages} pages, {n_scan} scan-only")
    for f in failures:
        print(f"  fail: {f}")
    for d in docs:
        if d.is_scan:
            print(f"  scan: {d.doc_id} ({len(d.pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
