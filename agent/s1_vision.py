"""S1 vision fallback — OCR for pages marked is_scan (too few extracted chars).

Renders each scan page via PyMuPDF and sends the PNG through the shared
LLMClient multimodal path. No separate OCR stack (tesseract/marker/docling) —
docs/STRATEGY.md / LINK-GUARDS-PLAN.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pymupdf as fitz
from pydantic import BaseModel, ConfigDict

from agent.llm import LLMClient
from agent.normalize import sanitize
from agent.progress import format_llm_label
from agent.schemas import IngestedDocument, PageText

_VISION_SYSTEM = """\
You are reading a scanned page from a Kazakhstan corporate loan / KYC / auditor PDF.
Transcribe all visible text as faithfully as possible. Preserve line breaks roughly.
If the page contains a TABLE (разовые статьи, обеспечение дочерних, ownership %),
transcribe every row: names, percentages, and dollar amounts exactly.
Do not summarize or invent text that is not visible. If the page is blank, return an empty string.
"""

_VISION_TABLE_SYSTEM = """\
You are OCR-transcribing a scanned TABLE page from auditor notes or KYC.
Output EVERY visible row and cell: counterparty names, dollar amounts, percentages,
and any materiality rule (e.g. разовые ≥ $300,000). Preserve numbers exactly.
Do not summarize. Empty page → empty string.
"""

# Long-side budget ≈1.15 MP / ~1590 vision tokens per page (API course revision).
# W5.5: final retry at 2500px for stubborn scan pages (P4 auditor table).
_MAX_SIDE_PX = 1500
_RETRY_MAX_SIDE_PX = 2000
_FINAL_RETRY_MAX_SIDE_PX = 2500
# Local OpenAI-compatible servers often hang on unsupported images; bound the wait
# so the pipeline can flag vision_failed and continue.
_VISION_CALL_TIMEOUT_S = 60.0


class PageTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


def image_cache_token(
    doc_sha256: str, page_number: int, max_side: int = _MAX_SIDE_PX, *, variant: str = ""
) -> str:
    """Stable cache token for a rendered page — params, not PNG bytes."""
    base = f"{doc_sha256}:p{page_number}:max_side={max_side}:png"
    return f"{base}:{variant}" if variant else base


def render_page_png(pdf_path: Path | str, page_number: int, max_side: int = _MAX_SIDE_PX) -> bytes:
    """Rasterize one 1-indexed PDF page to PNG with long side ≤ max_side."""
    with fitz.open(pdf_path) as pdf:
        page = pdf[page_number - 1]
        rect = page.rect
        long_side = max(rect.width, rect.height) or 1.0
        scale = min(1.0, max_side / long_side)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")


async def _transcribe_page(
    llm: LLMClient,
    doc: IngestedDocument,
    page: PageText,
    *,
    system: str,
    user: str,
    max_side: int,
    cache_dir: Path | None,
    use_cache: bool,
    variant: str,
) -> str:
    token = image_cache_token(doc.sha256, page.page_number, max_side, variant=variant)
    png = render_page_png(doc.path, page.page_number, max_side)
    label = format_llm_label(doc_id=doc.doc_id, extra=f"p{page.page_number}")
    result = await asyncio.wait_for(
        llm.extract(
            system,
            user,
            PageTranscript,
            cache_dir=cache_dir,
            use_cache=use_cache,
            label=label,
            images=[png],
            image_cache_token=token,
        ),
        timeout=_VISION_CALL_TIMEOUT_S,
    )
    return result.text or ""


async def ocr_scan_document(
    llm: LLMClient,
    doc: IngestedDocument,
    cache_dir: Path | None,
    use_cache: bool = True,
    vision_char_threshold: int = 50,
) -> tuple[IngestedDocument, list[str]]:
    """Replaces scan-page text with a vision transcription. Returns (doc, flags)."""
    if not any(p.is_scan for p in doc.pages):
        return doc, []

    flags: list[str] = []
    new_pages: list[PageText] = []
    for page in doc.pages:
        if not page.is_scan:
            new_pages.append(page)
            continue

        user = (
            f"Document {doc.doc_id} page {page.page_number} is a scanned page "
            f"(pre-OCR char_count={page.char_count}). Transcribe all visible text from the image, "
            f"including every table cell (names, %, $ amounts)."
        )
        raw = ""
        used_side = _MAX_SIDE_PX
        attempts: list[tuple[str, int, str]] = [
            (_VISION_SYSTEM, _MAX_SIDE_PX, ""),
            (_VISION_SYSTEM, _RETRY_MAX_SIDE_PX, ""),
            (_VISION_TABLE_SYSTEM, _RETRY_MAX_SIDE_PX, "table"),
            (_VISION_TABLE_SYSTEM, _FINAL_RETRY_MAX_SIDE_PX, "table_hi"),
        ]
        for system, side, variant in attempts:
            used_side = side
            try:
                raw = await _transcribe_page(
                    llm,
                    doc,
                    page,
                    system=system,
                    user=user,
                    max_side=side,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    variant=variant,
                )
            except Exception:  # noqa: BLE001 — keep page as scan; flag loudly
                raw = ""
            if len(sanitize(raw).strip()) >= vision_char_threshold:
                break

        normalized = sanitize(raw)
        char_count = len(normalized.strip())
        if char_count < vision_char_threshold:
            flags.append(
                f"VISION_PAGE_FAILED: doc {doc.doc_id} page {page.page_number} "
                f"transcript ~{char_count} chars (render max_side={used_side})"
            )
            flags.append(
                f"vision_failed: doc {doc.doc_id} page {page.page_number} "
                f"transcript ~{char_count} chars (render max_side={used_side})"
            )
        new_pages.append(
            PageText(
                page_number=page.page_number,
                raw_text=raw,
                normalized_text=normalized,
                char_count=char_count,
                is_scan=char_count < vision_char_threshold,
            )
        )
    return (
        IngestedDocument(doc_id=doc.doc_id, path=doc.path, sha256=doc.sha256, pages=new_pages),
        flags,
    )
