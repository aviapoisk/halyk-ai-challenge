"""Pre-flight smoke checks for a private dataset (VERIFY-PRIVATE §2).

No LLM. Run before the first battle-day agent.run — any FAIL must be fixed first.

Usage:
    python -m tools.smoke_private --dataset <DIR> [--cache .cache]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agent.txn_ids import parse_txn_scenario

# Expected open-set baseline columns (informational WARN baseline, not agent hardcoding).
_EXPECTED_COLS = (
    "txn_id",
    "date",
    "account_id",
    "counterparty",
    "description",
    "amount",
    "currency",
)
_REQUIRED_COLS = {"txn_id", "account_id", "amount"}
_TOP_LEVEL_KEYS = {"team", "contact_email", "model", "answers"}
_TXN_RE = re.compile(r"^TXN-(?P<rest>.+)-(?P<seq>\d+)$")
_ACC_RE = re.compile(r"^ACC-(?P<digits>\d+)$")
_CLAUSE_RE = re.compile(r"^\d+\.\d+$")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_YEAR_IN_NAME = re.compile(r"(20\d{2})")
_VISION_CHAR_THRESHOLD = 50
_VISION_CANDIDATE_CAP = 20
_LATIN_HEADER_EXAMPLES = 5


@dataclass
class CheckResult:
    level: str  # PASS | WARN | FAIL | INFO
    code: str
    msg: str

    def __str__(self) -> str:
        return f"{self.level}  [{self.code}] {self.msg}"


@dataclass
class CheckBag:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, level: str, code: str, msg: str) -> None:
        self.results.append(CheckResult(level, code, msg))

    def pass_(self, code: str, msg: str) -> None:
        self.add("PASS", code, msg)

    def warn(self, code: str, msg: str) -> None:
        self.add("WARN", code, msg)

    def fail(self, code: str, msg: str) -> None:
        self.add("FAIL", code, msg)

    def info(self, code: str, msg: str) -> None:
        self.add("INFO", code, msg)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.level == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.results if r.level == "WARN")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Private-dataset pre-flight smoke checks (VERIFY-PRIVATE §2)"
    )
    p.add_argument("--dataset", required=True, type=Path, help="dataset root directory")
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache"),
        help="LLM/parsed cache dir to inspect for open→private leak risk (default: .cache)",
    )
    return p


def _read_ledger_rows(ledger: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(ledger, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    return cols, rows


def _year_from_date(raw: str) -> int | None:
    s = (raw or "").strip()
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        if 1990 <= y <= 2100:
            return y
    m = re.search(r"(20\d{2})", s)
    if m:
        return int(m.group(1))
    return None


def check_ledger(bag: CheckBag, dataset: Path, ledger: Path) -> list[dict[str, str]]:
    print("=== 1. ledger ===")
    name_globs = sorted(dataset.resolve().rglob("*ledger*.csv"))
    bag.info(
        "ledger.glob",
        f"*ledger*.csv matches: {[p.name for p in name_globs] or '(none)'}",
    )
    bag.pass_("ledger.found", f"discovered ledger={ledger} ({ledger.stat().st_size} bytes)")

    if "2025" in ledger.name:
        bag.warn(
            "ledger.year_in_name",
            f"ledger name still contains 2025 ({ledger.name}) — confirm this is the private set",
        )

    if name_globs:
        names = {p.resolve() for p in name_globs}
        if ledger.resolve() not in names:
            bag.warn(
                "ledger.glob_mismatch",
                f"discovered ledger {ledger.name} is not among *ledger*.csv matches "
                f"{[p.name for p in name_globs]}",
            )
        elif len(name_globs) > 1:
            bag.warn(
                "ledger.multiple",
                f"multiple *ledger*.csv files; pipeline uses largest by size → {ledger.name}",
            )

    cols, rows = _read_ledger_rows(ledger)
    col_set = {c.strip() for c in cols}
    missing_req = _REQUIRED_COLS - col_set
    if missing_req:
        bag.fail("ledger.cols_required", f"missing required columns: {sorted(missing_req)}")
    else:
        bag.pass_("ledger.cols_required", f"required columns present: {sorted(_REQUIRED_COLS)}")

    expected = set(_EXPECTED_COLS)
    missing_opt = expected - col_set
    extra = col_set - expected
    if missing_opt:
        bag.warn("ledger.cols_expected", f"missing expected columns: {sorted(missing_opt)}")
    if extra:
        bag.warn("ledger.cols_extra", f"unexpected columns: {sorted(extra)}")
    if not missing_opt and not extra and not missing_req:
        bag.pass_("ledger.cols", f"columns match baseline 7: {list(cols)}")

    years: set[int] = set()
    empty_dates = 0
    for row in rows:
        y = _year_from_date(row.get("date", ""))
        if y is None:
            empty_dates += 1
        else:
            years.add(y)

    bag.info("ledger.rows", f"rows={len(rows)}")
    if not years:
        bag.warn("ledger.dates", f"no parseable years in date column (empty/unparsed={empty_dates})")
    else:
        bag.pass_(
            "ledger.dates",
            f"date years min={min(years)} max={max(years)} (unparsed={empty_dates})",
        )
        name_years = {int(m) for m in _YEAR_IN_NAME.findall(ledger.name)}
        if name_years and years.isdisjoint(name_years):
            bag.warn(
                "ledger.year_vs_name",
                f"date years {sorted(years)} do not overlap year(s) in filename {sorted(name_years)}",
            )

    for r in bag.results:
        if r.code.startswith("ledger."):
            print(r)
    print()
    return rows


def check_ids(bag: CheckBag, rows: list[dict[str, str]]) -> None:
    print("=== 2. txn_id / account_id ===")
    n = len(rows)
    if n == 0:
        bag.fail("ids.empty", "ledger has zero rows")
        for r in bag.results:
            if r.code.startswith("ids."):
                print(r)
        print()
        return

    acc_prefix = 0
    digit_lens: Counter[int] = Counter()
    non_acc_examples: list[str] = []
    txn_ok = 0
    scenarios: set[str] = set()
    bad_txn: list[str] = []

    for row in rows:
        acc = (row.get("account_id") or "").strip()
        m_acc = _ACC_RE.match(acc)
        if m_acc:
            acc_prefix += 1
            digit_lens[len(m_acc.group("digits"))] += 1
        elif len(non_acc_examples) < 5:
            non_acc_examples.append(acc or "(empty)")

        txn = (row.get("txn_id") or "").strip()
        m_txn = _TXN_RE.match(txn)
        if m_txn:
            txn_ok += 1
            scenarios.add(m_txn.group("rest"))
        elif len(bad_txn) < 5:
            bad_txn.append(txn or "(empty)")

    acc_rate = acc_prefix / n
    if acc_rate >= 0.99:
        bag.pass_("ids.acc_prefix", f"ACC- prefix on {acc_prefix}/{n} account_id ({acc_rate:.1%})")
    elif acc_rate >= 0.5:
        bag.warn(
            "ids.acc_prefix",
            f"ACC- prefix only on {acc_prefix}/{n} ({acc_rate:.1%}); examples={non_acc_examples}",
        )
    else:
        bag.warn(
            "ids.acc_prefix",
            f"ACC- prefix rare: {acc_prefix}/{n} ({acc_rate:.1%}); examples={non_acc_examples}",
        )

    lens = sorted(digit_lens)
    bag.info("ids.acc_digit_lens", f"digit lengths after ACC-: {dict(digit_lens)}")
    if lens and set(lens) != {4}:
        bag.warn(
            "ids.acc_digit_len",
            f"digit lengths {lens} (legacy anchor assumes 4; runtime uses make_anchor_pattern)",
        )
    elif lens == [4]:
        bag.pass_("ids.acc_digit_len", "all ACC- ids have 4 digits")

    txn_rate = txn_ok / n
    bag.info(
        "ids.txn_format",
        f"TXN-<sid>-NNNN match {txn_ok}/{n} ({txn_rate:.1%}); "
        f"unique scenarios={len(scenarios)} {sorted(scenarios)[:20]}"
        + ("…" if len(scenarios) > 20 else ""),
    )
    if txn_rate < 0.5:
        bag.fail(
            "ids.txn_format",
            f"txn_id match-rate {txn_rate:.1%} < 50% — account↔scenario binding will break; "
            f"bad examples={bad_txn}",
        )
    elif bad_txn:
        bag.warn("ids.txn_nonmatch", f"non-matching txn_id examples={bad_txn}")
        bag.pass_("ids.txn_format", f"txn_id format OK ({txn_rate:.1%} match)")
    else:
        bag.pass_("ids.txn_format", f"all txn_id match TXN-<sid>-NNNN ({txn_ok})")

    for r in bag.results:
        if r.code.startswith("ids."):
            print(r)
    print()


def check_template(bag: CheckBag, template_path: Path) -> None:
    print("=== 3. submission_template ===")
    try:
        data = json.loads(template_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        bag.fail("template.parse", f"cannot parse {template_path}: {exc}")
        for r in bag.results:
            if r.code.startswith("template."):
                print(r)
        print()
        return

    bag.pass_("template.found", f"template={template_path}")
    top = set(data.keys()) if isinstance(data, dict) else set()
    missing = _TOP_LEVEL_KEYS - top
    extra = top - _TOP_LEVEL_KEYS
    if missing:
        bag.warn("template.top_missing", f"missing top-level keys: {sorted(missing)}")
    if extra:
        bag.warn("template.top_extra", f"unexpected top-level keys: {sorted(extra)}")
    if not missing and not extra:
        bag.pass_("template.top", f"top-level keys OK: {sorted(top)}")

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict) or not answers:
        bag.fail("template.answers", "answers missing or empty")
        for r in bag.results:
            if r.code.startswith("template."):
                print(r)
        print()
        return

    scenarios = sorted(answers.keys())
    clause_keys: set[str] = set()
    non_numeric: list[str] = []
    for sid, clauses in answers.items():
        if not isinstance(clauses, dict):
            bag.warn("template.clauses_type", f"{sid}: clauses is not an object")
            continue
        for ck in clauses:
            clause_keys.add(ck)
            if not _CLAUSE_RE.match(ck):
                non_numeric.append(f"{sid}/{ck}")

    bag.pass_(
        "template.scenarios",
        f"scenarios={len(scenarios)}: {scenarios}",
    )
    bag.info("template.clauses", f"unique clause keys: {sorted(clause_keys)}")
    non_6x = sorted(k for k in clause_keys if not k.startswith("6."))
    if non_6x:
        bag.warn("template.non_6x", f"clause keys not starting with 6.: {non_6x}")
    else:
        bag.pass_("template.clause_prefix", "all clause keys are 6.x")
    if non_numeric:
        bag.warn("template.clause_shape", f"clause keys not N.M: {non_numeric[:20]}")
    else:
        bag.pass_("template.clause_shape", "all clause keys look like N.M")

    for r in bag.results:
        if r.code.startswith("template."):
            print(r)
    print()


def check_template_ledger_align(
    bag: CheckBag, rows: list[dict[str, str]], template_path: Path
) -> None:
    """FAIL if a template scenario has zero ledger rows after txn→sid resolve."""
    print("=== 3b. template ↔ ledger (resolved) ===")
    try:
        data = json.loads(template_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        bag.fail("align.template_parse", f"cannot parse {template_path}: {exc}")
        for r in bag.results:
            if r.code.startswith("align."):
                print(r)
        print()
        return

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict) or not answers:
        bag.fail("align.answers", "answers missing or empty — cannot align")
        for r in bag.results:
            if r.code.startswith("align."):
                print(r)
        print()
        return

    known = set(answers.keys())
    rows_by_sid: Counter[str] = Counter()
    accounts_by_sid: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        sid = parse_txn_scenario(row.get("txn_id", ""), known)
        if sid is None or sid not in known:
            continue
        rows_by_sid[sid] += 1
        acc = (row.get("account_id") or "").strip() or "(empty)"
        accounts_by_sid[sid][acc] += 1

    missing = sorted(known - set(rows_by_sid))
    if missing:
        bag.fail(
            "align.missing_ledger",
            f"template scenarios with 0 ledger rows after resolve: {missing}",
        )
    else:
        bag.pass_(
            "align.coverage",
            f"all {len(known)} template scenarios have ≥1 ledger row after resolve",
        )

    for sid in sorted(known):
        bag.info(
            "align.sid_accounts",
            f"{sid}: rows={rows_by_sid.get(sid, 0)} accounts={dict(accounts_by_sid.get(sid, {}))}",
        )

    for r in bag.results:
        if r.code.startswith("align."):
            print(r)
    print()


def _pdf_page_stats(path: Path, threshold: int) -> tuple[int, int, str]:
    """Return (n_pages, n_sparse_pages, header_sample)."""
    import pymupdf as fitz

    n_pages = 0
    n_sparse = 0
    header = ""
    with fitz.open(path) as doc:
        n_pages = doc.page_count
        for i, page in enumerate(doc):
            text = page.get_text() or ""
            if i == 0:
                header = text[:500]
            if len(text.strip()) < threshold:
                n_sparse += 1
    return n_pages, n_sparse, header


def check_documents(bag: CheckBag, documents: Path) -> list[tuple[Path, str]]:
    """Return list of (pdf_path, header_text) for language check."""
    print("=== 4. documents ===")
    if not documents.is_dir():
        bag.fail("docs.dir", f"documents dir missing: {documents}")
        for r in bag.results:
            if r.code.startswith("docs."):
                print(r)
        print()
        return []

    files = [p for p in documents.iterdir() if p.is_file()]
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    non_pdf = [p for p in files if p.suffix.lower() != ".pdf"]
    total_bytes = sum(p.stat().st_size for p in files)
    total_kb = total_bytes / 1024.0
    token_est = total_kb * 100

    bag.pass_(
        "docs.counts",
        f"files={len(files)} pdf={len(pdfs)} non_pdf={len(non_pdf)} "
        f"size={total_kb:.0f} KB (~{token_est:.0f} tokens @ 100 tok/KB)",
    )
    bag.info(
        "docs.time_hint",
        "rough time ≈ T_open × (private_KB / open_KB); measure T_open on cold open run "
        f"(this corpus ~{total_kb:.0f} KB)",
    )
    if non_pdf:
        by_ext: dict[str, int] = defaultdict(int)
        for p in non_pdf:
            by_ext[p.suffix.lower() or "(none)"] += 1
        bag.warn("docs.non_pdf", f"non-PDF files by ext: {dict(by_ext)}")

    try:
        import pymupdf as fitz  # noqa: F401
    except ImportError:
        bag.fail(
            "docs.pymupdf",
            "pymupdf not installed in this interpreter — run via .venv/bin/python",
        )
        for r in bag.results:
            if r.code.startswith("docs."):
                print(r)
        print()
        return []

    vision_candidates: list[tuple[str, int, int]] = []
    headers: list[tuple[Path, str]] = []
    parse_fail = 0
    for path in sorted(pdfs, key=lambda p: p.name):
        try:
            n_pages, n_sparse, header = _pdf_page_stats(path, _VISION_CHAR_THRESHOLD)
        except Exception as exc:  # noqa: BLE001
            parse_fail += 1
            if parse_fail <= 5:
                bag.warn("docs.pdf_open", f"cannot open {path.name}: {exc}")
            continue
        headers.append((path, header))
        if n_sparse > 0 or n_pages == 0:
            vision_candidates.append((path.name, n_sparse, n_pages))

    bag.info(
        "docs.vision_threshold",
        f"vision candidate = page with <{_VISION_CHAR_THRESHOLD} chars (S1 default)",
    )
    if vision_candidates:
        shown = vision_candidates[:_VISION_CANDIDATE_CAP]
        bag.warn(
            "docs.vision_candidates",
            f"{len(vision_candidates)} PDF(s) with sparse page(s); "
            f"showing {len(shown)}: "
            + ", ".join(f"{n}({sp}/{pg} sparse)" for n, sp, pg in shown),
        )
    else:
        bag.pass_("docs.vision_candidates", "no PDF with pages below vision char threshold")
    if parse_fail:
        bag.warn("docs.pdf_open_count", f"{parse_fail} PDF(s) failed to open")

    for r in bag.results:
        if r.code.startswith("docs."):
            print(r)
    print()
    return headers


def check_languages(bag: CheckBag, headers: list[tuple[Path, str]]) -> None:
    print("=== 5. languages ===")
    if not headers:
        bag.warn("lang.none", "no PDF headers to inspect")
        for r in bag.results:
            if r.code.startswith("lang."):
                print(r)
        print()
        return

    latin_docs = 0
    examples: list[str] = []
    for path, header in headers:
        if _LATIN_WORD.search(header):
            latin_docs += 1
            if len(examples) < _LATIN_HEADER_EXAMPLES:
                first_line = next(
                    (ln.strip() for ln in header.splitlines() if ln.strip()),
                    header[:80].replace("\n", " "),
                )
                examples.append(f"{path.name}: {first_line[:100]}")

    rate = latin_docs / len(headers)
    bag.info(
        "lang.latin_header",
        f"latin in header: {latin_docs}/{len(headers)} ({rate:.1%})",
    )
    for ex in examples:
        bag.info("lang.example", ex)
    if rate < 0.05:
        bag.warn("lang.low_latin", "very few latin headers — unexpected if EN docs confirmed")
    else:
        bag.pass_("lang.latin_ok", "latin headers present (EN docs expected)")

    for r in bag.results:
        if r.code.startswith("lang."):
            print(r)
    print()


def check_fx(bag: CheckBag, rows: list[dict[str, str]]) -> None:
    print("=== 6. currencies / FX ===")
    if not rows:
        bag.warn("fx.empty", "no ledger rows for currency check")
        for r in bag.results:
            if r.code.startswith("fx."):
                print(r)
        print()
        return

    counts: Counter[str] = Counter()
    for row in rows:
        cur = (row.get("currency") or "").strip() or "(empty)"
        counts[cur] += 1

    bag.info("fx.counts", f"currency counts: {dict(sorted(counts.items()))}")
    non_usd = {c: n for c, n in counts.items() if c != "USD"}
    if non_usd:
        bag.warn(
            "fx.non_usd",
            f"non-USD rows={sum(non_usd.values())} currencies={non_usd}; "
            "need in-document FX source (implied pair / auditor note), not an external table",
        )
    else:
        bag.pass_("fx.usd_only", "all ledger rows are USD")

    for r in bag.results:
        if r.code.startswith("fx."):
            print(r)
    print()


def check_env_cache(bag: CheckBag, cache_dir: Path) -> None:
    print("=== 7. .env / cache ===")
    load_dotenv()

    key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    provider = os.environ.get("AGENT_PROVIDER", "")
    model = os.environ.get("AGENT_MODEL", "")
    team = os.environ.get("TEAM_NAME", "")
    email = os.environ.get("CONTACT_EMAIL", "")
    concurrency = os.environ.get("AGENT_LLM_CONCURRENCY", "")

    if key.strip():
        bag.pass_("env.api_key", "API key present (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
    else:
        bag.fail("env.api_key", "no ANTHROPIC_API_KEY / OPENAI_API_KEY")

    bag.info(
        "env.meta",
        f"PROVIDER={provider or '(unset)'} MODEL={model or '(unset)'} "
        f"CONCURRENCY={concurrency or '(unset→12)'} "
        f"TEAM={'set' if team.strip() else '(empty)'} "
        f"EMAIL={'set' if email.strip() else '(empty)'}",
    )
    if not model.strip():
        bag.warn("env.model", "AGENT_MODEL unset")
    if not team.strip():
        bag.warn("env.team", "TEAM_NAME unset")
    if not email.strip():
        bag.warn("env.email", "CONTACT_EMAIL unset")

    bag.warn(
        "env.tpm",
        "no TPM env var — verify API key tier TPM/RPM manually before private window (×2–3 corpus)",
    )

    llm_dir = cache_dir / "llm"
    parsed_dir = cache_dir / "parsed"
    llm_n = sum(1 for _ in llm_dir.glob("*.json")) if llm_dir.is_dir() else 0
    parsed_n = sum(1 for _ in parsed_dir.glob("*.json")) if parsed_dir.is_dir() else 0
    bag.info(
        "cache.stats",
        f"cache={cache_dir.resolve()} llm_entries={llm_n} parsed_entries={parsed_n}",
    )
    bag.info(
        "cache.namespace",
        "pipeline does NOT namespace cache by dataset — shared .cache can mix open/private",
    )
    if llm_n > 0:
        bag.warn(
            "cache.nonempty",
            f"{llm_n} llm cache entries present; for private run use "
            f"`--cache .cache/private` (empty) or `--no-cache` to avoid open-set leak class",
        )
    else:
        bag.pass_("cache.empty_llm", "llm cache empty or missing — low open→private leak risk")

    for r in bag.results:
        if r.code.startswith("env.") or r.code.startswith("cache."):
            print(r)
    print()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bag = CheckBag()

    from agent.dataset import discover_dataset

    try:
        paths = discover_dataset(args.dataset)
    except FileNotFoundError as exc:
        print("=== discover ===")
        print(f"FAIL  [discover] {exc}")
        print("\n1 fail, 0 warn")
        return 1

    print(f"dataset={args.dataset.resolve()}")
    print(f"ledger={paths.ledger}")
    print(f"template={paths.template}")
    print(f"documents={paths.documents}")
    print()

    rows = check_ledger(bag, args.dataset, paths.ledger)
    check_ids(bag, rows)
    check_template(bag, paths.template)
    check_template_ledger_align(bag, rows, paths.template)
    headers = check_documents(bag, paths.documents)
    check_languages(bag, headers)
    check_fx(bag, rows)
    check_env_cache(bag, args.cache)

    print("=== summary ===")
    print(f"{bag.n_fail} fail, {bag.n_warn} warn")
    return 1 if bag.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
