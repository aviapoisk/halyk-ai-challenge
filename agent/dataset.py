"""Discover dataset layout without hardcoded filenames (year in ledger name, etc.)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

_LEDGER_COLS = {"txn_id", "account_id", "amount"}
_DOC_EXTS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".md"}


@dataclass(frozen=True)
class DatasetPaths:
    ledger: Path
    template: Path
    documents: Path


def _looks_like_ledger(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = next(csv.reader(f), [])
    except OSError:
        return False
    return _LEDGER_COLS <= {c.strip().lower() for c in header}


def _looks_like_template(path: Path) -> bool:
    """Template = answers → clauses → cells with empty status.

    Filled status means ground_truth or a prior submission — not a template.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    answers = data.get("answers")
    if not isinstance(answers, dict) or not answers:
        return False
    for clauses in answers.values():
        if not isinstance(clauses, dict) or not clauses:
            return False
        for cell in clauses.values():
            if not isinstance(cell, dict) or "status" not in cell or cell["status"] is not None:
                return False
    return True


def discover_dataset(root: Path) -> DatasetPaths:
    root = root.resolve()
    ledgers = [p for p in root.rglob("*.csv") if _looks_like_ledger(p)]
    templates = [p for p in root.rglob("*.json") if _looks_like_template(p)]
    if not ledgers:
        raise FileNotFoundError(f"в {root} нет CSV с колонками {sorted(_LEDGER_COLS)}")
    if not templates:
        raise FileNotFoundError(f"в {root} нет submission-шаблона с пустыми ячейками")
    ledger = max(ledgers, key=lambda p: p.stat().st_size)
    template = sorted(templates)[0]
    dirs = [root, *(p for p in root.rglob("*") if p.is_dir())]
    documents = max(
        dirs,
        key=lambda d: sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in _DOC_EXTS),
    )
    return DatasetPaths(ledger=ledger, template=template, documents=documents)
