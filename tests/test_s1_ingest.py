"""S1 ingest: atomic cache write + empty-dir / order smoke (no LLM)."""

from __future__ import annotations

from pathlib import Path

from agent.s1_ingest import (
    _SPARSE_IMAGE_CHAR_CEILING,
    _write_parsed_cache,
    ingest_directory,
)
from agent.s1_vision import image_cache_token
from agent.schemas import IngestedDocument, PageText


def test_write_parsed_cache_atomic(tmp_path: Path):
    cache_file = tmp_path / "parsed" / "abc.json"
    doc = IngestedDocument(
        doc_id="x",
        path="x.pdf",
        sha256="abc",
        pages=[
            PageText(
                page_number=1,
                raw_text="hi",
                normalized_text="hi",
                char_count=2,
                is_scan=False,
            )
        ],
    )
    _write_parsed_cache(cache_file, doc)
    assert cache_file.exists()
    assert not cache_file.with_suffix(".json.tmp").exists()
    loaded = IngestedDocument.model_validate_json(cache_file.read_text(encoding="utf-8"))
    assert loaded.doc_id == "x"


def test_ingest_directory_empty(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    docs, failures = ingest_directory(docs_dir, cache_dir=tmp_path / "cache")
    assert docs == []
    assert failures == []


def test_ingest_plaintext_and_unsupported(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "note.txt").write_text("hello ACC-1", encoding="utf-8")
    (docs_dir / "noise.docx").write_bytes(b"PK")
    docs, failures = ingest_directory(docs_dir, cache_dir=tmp_path / "cache", max_workers=1)
    assert len(docs) == 1
    assert docs[0].doc_id == "note"
    assert any("docx" in f for f in failures)


def test_ingest_bad_pdf_does_not_raise(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "broken.pdf").write_bytes(b"not-a-pdf")
    docs, failures = ingest_directory(docs_dir, cache_dir=tmp_path / "cache", max_workers=1)
    assert docs == []
    assert failures
    assert "broken.pdf" in failures[0]


def test_sparse_image_ceiling_and_table_cache_variant():
    assert _SPARSE_IMAGE_CHAR_CEILING == 200
    base = image_cache_token("abc", 4, 2000)
    table = image_cache_token("abc", 4, 2000, variant="table")
    assert base != table
    assert table.endswith(":table")
