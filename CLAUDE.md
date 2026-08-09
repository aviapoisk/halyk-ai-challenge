# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Prep / submission workspace for the **Halyk AI Challenge** hackathon (official task spec: `data/agentic-bank-public/CASE.ru.md` / `CASE.kz.md`). Task: build an AI agent that reads unstructured corporate loan documents (PDFs) plus a shared transaction ledger, and for every `(borrower scenario_id, covenant clause)` pair outputs `status` (COMPLIANT/BREACH), `actual` (the metric value), and `evidence_txn_id` (the single transaction whose treatment flips the verdict, or `null`).

The battle path lives under `agent/` (`python -m agent.run`); companion tools under `tools/`; fast unit tests under `tests/`. Binding engineering rules for the pipeline are in this file (sections below).

## Battle-day constraints (read before touching anything)

- Deadline: private dataset drops 2026-08-09 11:00 (Astana), submission closes 14:00 same day. Submission scored on the **last** attempt uploaded, not the first.
- Organizers rerun the solution with their own API keys during judging — the repo must include a working `.env.example` + README and be runnable end-to-end without manual intervention.
- **Manual/human-edited answers are disqualifying.** Any HITL / TUI override path is forbidden on the live run; `tools/dev.py` is read-only triage and must never be used to edit answers.
- `submission.json` keys must byte-exactly match `submission_template.json` — never add/remove/rename keys. Cells are sanitized then validated; the submission file is **always written** (absence of a file is worse than a flawed cell).

## Source materials in this repo

1. `data/agentic-bank-public/CASE.ru.md` / `CASE.kz.md` — official task spec (ground truth for requirements).
2. `data/agentic-bank-public/submission_template.json` / `ground_truth.json` / `master_ledger_2025.csv` / `documents/` — the open sample dataset; `ground_truth.json` is a dev oracle for the open set only (`scenarios.*.covenants` shape — scorer accepts it).
3. This file + `README.md` — architecture, CLI contract, and engineering constraints for code changes.
4. `.env.example` — provider / model / team fields filled by the pipeline (never hand-edit `submission.json`).

Internal planning notes (`docs/STRATEGY.md`, `docs/FINDINGS-*`, `docs/TESTING-PLAN.md`, early `refs/*` brainstorms) are **not shipped in this tree**. Do not invent paths to them; treat the rules below as authoritative for agent code.

## Scoring model (drives all design decisions)

Per cell: `status` worth 0.50 (wrong verdict zeroes the whole cell), `actual` worth 0.30 (linear decay to zero at 5% relative error), `evidence_txn_id` worth 0.20 (decays together with `actual` when the correct evidence is `null` — most open-GT cells have null evidence, so number accuracy is worth 0.50 there in practice). Cell difficulty weights are unpublished. Tie-break favors the earlier best submission. This means: never hedge/average `actual` between competing hypotheses — pick the single best guess, since the scale is linear.

## Architecture (accepted design)

A deterministic-first pipeline, one CLI, no async-framework/orchestrator overengineering (a `asyncio.gather` + semaphore pool is sufficient for ~12 borrowers / ~200 small PDFs; concurrency comes from `AGENT_LLM_CONCURRENCY` / `--profile`):

- **S1 Ingestion** — PyMuPDF for text-layer PDFs (covers 199/200 open-set docs); vision/LLM fallback reserved for true scans only (1/200 in the open set). Every parse is cached by SHA of file content.
- **S2 Classification & linking** — LLM classifies each doc's type/account/year/validity. `account_id → scenario_id` is derived **dynamically** from the ledger's `txn_id` prefix (`TXN-<scenario_id>-NNNN`) — never hardcode this mapping. Entity linking uses `ACC-\d{4}` regex anchors (present in only ~1/3 of docs — normalize text first, letter-spacing breaks naive regex) plus a second channel matching the exact normalized company name from the page-1 header (covers most of the rest); fuzzy name matching is deliberately avoided (near-duplicate company names in the corpus make it dangerous).
- **S3 Case compilation** — LLM extracts covenant formulas/thresholds/carve-outs, auditor reclassifications, related-party relationships (KYC), and FX rates, each with a source citation.
- **S4 Deterministic engine** — all arithmetic (filtering, currency conversion, reclassification, threshold comparison) is done in pandas/Python, not by the LLM. Principle: "LLM extracts meaning, Python computes." Includes a generic/tree-fallback for covenant types not seen before (`--engine enum|shadow|tree`).
- **S5 Evidence** — candidate transactions are flip-tested: recompute the metric excluding the candidate and check whether the verdict flips. Also usable as an evidence *generator*, not just a verifier.
- **S6 Confidence report + validator** — sanitize cells, validate against `submission_template.json`, always write atomically; confidence/triage report rather than a manual-override UI.

Caching: two-level sidecar file cache — parsed-document cache keyed by content hash, LLM-response cache keyed by hash of (model + effort + system prompt + doc text + schema), under `.cache/`. Supports `--no-cache` for live reruns (organizers' own keys).

## Known dataset traps (confirmed in the open set)

Outdated/superseded contract versions in the corpus; noise documents containing plausible-looking numbers; one genuine scanned PDF; EUR transactions with in-document FX rates; formulas requiring cross-document data (e.g. Capex sourced from a parent company doc, EBITDA from the borrower doc); auditor reclassifications that change which ledger lines count; related-party matching via KYC aliases; carve-out clauses where a BREACH-looking number is still contractually COMPLIANT; ~800 noise ledger rows on unrelated `ACC-9xxx` accounts mixed into the 1473-row ledger with the ~673 real target rows.

Explicitly rejected as not present in this dataset (do not build for them): MCC codes, BIN/IIN checksum validation, МРП/МЗП indexed limits, KZT currency, "BREACH implies evidence_txn_id must be set" (false — 8 open-GT BREACH cells have null evidence).

## Engineering rules for the pipeline code

Binding rules for all agent code written in this repo. Several are disqualification-related, not style.

### CLI contract (fixed — tests and README depend on it)

```
python -m agent.run --dataset <DIR> --out <FILE.json> [--profile default] [--cache .cache/] [--seed 42]
```

- `<DIR>` is *any* folder containing `master_ledger_*.csv`, `documents/`, `submission_template.json`. One process, no interactive input.
- If the API dies mid-run, still write a **valid submission with every cell filled** (best deterministic guess + flag in the confidence report). An empty cell scores the same as a wrong one; a crash without a file scores worse than both.
- Companion CLIs: `tools/validate.py` (contract validator — cells are sanitized then validated; the submission file is always written), `tools/score.py` (official formula incl. actual decay and the evidence-null rule, lenient + strict modes; open set only). Optional diagnostics land in `logs/` next to CWD or under `<out.parent>/logs/` when `--out` is not in CWD (e.g. `runs/<id>/logs/`).

### Determinism (organizers rerun the code and must get identical bytes)

- Every LLM call: strict Pydantic schema, routed through **one** client wrapper with the sidecar cache (key = SHA-256 of model + prompt + content + schema). No scattered SDK calls. OpenAI-compatible path sets `temperature=0`; Anthropic structured-output path omits temperature (API rejects it).
- No timestamps inside outputs, no unseeded randomness, no reliance on dict/filesystem ordering — sort everything that reaches the output file.
- Two consecutive warm-cache runs should produce byte-identical `submission.json`, except repair/self-consistency paths that intentionally call with `use_cache=False` (`agent/s3_extract.py` threshold/reclass repair, `agent/run.py` self-consistency pass).
- Local cache in `.cache/` (gitignored). A curated reproducibility cache under `data/cache/` is allowed by `.gitignore` but is not shipped — live runs use `.cache/` or `--no-cache`.
- `submission.json` is written atomically (tmp → rename) after sanitize; validation failures are reported but do not block the write.

### No hardcoding (the private set changes every identifier)

- Scenario ids and clause numbers come **only** from `submission_template.json` (`answers` keys); `account_id ↔ scenario_id` only from the ledger `txn_id` prefix at runtime; covenant parameters only from documents. Borrower/document/clause counts are variables.
- Open-set literals (`P5`, `ACC-7805`, `f3fa6d20c8a1.pdf`, thresholds, FX rates) may appear in `tests/` fixtures only — never in agent code.
- Model names live in env/config, never inline.

### LLM discipline: LLM extracts, Python decides

- Extraction is **per-document**; the borrower case is assembled in Python from per-document JSON — context is bounded by one document, not one case. (A whole-case "wide" profile may exist behind a flag, never as the default path.)
- Ambiguous ledger-row categorization: batched LLM labeling with caching — never per-row calls in a loop.
- Regex/anchor matching only on **normalized** text (collapse letter-spacing, NBSP, quote variants) — naive matching silently loses 12 of 66 `ACC-\d{4}` anchors, including auditor docs whose absence flips verdicts. Citations are returned from the original, un-normalized text.
- Entity binding: exact match on the normalized full legal name, plus ACC anchors. Fuzzy matching may only *suggest* candidates in the confidence report — the corpus contains near-duplicate names of different companies (`Shymkent Refinery Services JSC` vs `Shymkent Refinery JSC`).
- Contract validity is decided by the validity dates in the document text covering the covenant period; the «НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ» stamp is a secondary signal only (organizer-confirmed).

### Numbers

- Full precision internally; round to 2 decimals **only at output**. `actual` = abs(metric) as a plain number (`9.45`, not `"9.45x"`). Zero points at 5% relative error means 0.002 absolute for a 0.04 ratio — no float shortcuts near thresholds.
- Never average competing hypotheses for `actual`: the decay is linear, so averaging never beats committing to the better-documented reading.
- FX rates only from the borrower's own documents; a missing rate is a loud flag, never a silent skip or an external lookup.

### Evidence

- The engine's flip-test decides: a candidate is evidence iff recomputing with it excluded/reclassified flips the verdict **and** it is the only such transaction; zero or several → `null`. Document-named candidates are a second vote; on conflict, submit the flip-test-confirmed value and flag the cell.
- Never implement "BREACH ⇒ evidence required" (false on 8 open-GT cells). "COMPLIANT ⇒ null" is a soft warning, never an auto-fix.

### Definition of done for any change

1. Fast unit tests pass (no LLM, run on every commit).
2. Full run on the open set + `tools/score.py` vs GT; per-cell diff reviewed.
3. Every GT divergence gets a journalled diagnosis (symptom / diagnosis / fix); a fix must be justified by document text — "raises the score but only affects P6" without a document basis is a revert candidate.
4. Frozen holdout scenarios **P4 / P7 / B4**: never tune against them; check them roughly once a day.
5. Prefer fixes justified by document text over score-chasing on a single open-set borrower. Do not claim 36/36 without a full scored run.

### Prohibitions

- No manual editing of answers/`submission.json`, ever; no HITL/override path importable from `agent.run` (dev-only triage tools live outside the battle path). `team`/`contact_email`/`model` are filled by the pipeline from config.
- Do not build or resurrect: MCC/BIN/МРП logic, prebuilt KZT support, orchestration frameworks (`asyncio.gather` + semaphore is the ceiling), an OCR subsystem (a per-page "too few chars → vision" rule only), TUI verdict overrides. An OpenAI-compatible local-server branch exists in `agent/llm.py` for offline/dev smoke tests only — battle default is the cloud provider from config/env.
- Dependency budget: `pymupdf`, `pandas`, `pydantic`, provider SDK, `tenacity`, `pytest` (+ `textual` only for `tools/dev.py`). Anything beyond needs a one-line written justification in the commit message.

### Code quality is scored

Organizers review top teams' code by hand: README with the rerun/cache story, `.env.example`, type hints on public functions, `trace.json` per run, no dead code, nothing that even looks like manual answer editing. Write for that reviewer.
