"""Pydantic models shared across the pipeline (S1-S6).

S3 fact schemas below are populated by agent/llm.py's structured-output calls,
so every schema meant to be sent as an LLM output_config.format schema sets
`model_config = ConfigDict(extra="forbid")` (required for `additionalProperties:
false` in the generated JSON schema).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.json_schema import SkipJsonSchema


class PageText(BaseModel):
    page_number: int
    raw_text: str
    normalized_text: str
    char_count: int
    is_scan: bool


class IngestedDocument(BaseModel):
    doc_id: str  # filename stem, e.g. "04eee2e9ba8c"
    path: str
    sha256: str
    pages: list[PageText]

    @property
    def raw_text(self) -> str:
        return "\n".join(p.raw_text for p in self.pages)

    @property
    def normalized_text(self) -> str:
        return "\n".join(p.normalized_text for p in self.pages)

    @property
    def is_scan(self) -> bool:
        """A document is treated as a scan if every page is below the text-density threshold."""
        return bool(self.pages) and all(p.is_scan for p in self.pages)


class DocumentLinkage(BaseModel):
    doc_id: str
    account_anchor: str | None = None  # e.g. "ACC-7805", from regex anchor
    company_name: str | None = None  # normalized full legal name from the page-1 header
    scenario_id: str | None = None  # resolved via account_anchor -> ledger, or company_name bootstrap
    link_channel: str | None = None  # "account_anchor" | "company_name" | None
    # Set only when scenario_id is None (LINK-GUARDS G3):
    # no_anchor | name_not_bootstrapped | name_ambiguous | anchor_unknown_account | anchor_ambiguous
    unlink_reason: str | None = None


# --- S3: per-document fact extraction (LLM output_config.format schemas) ---


CategoryName = Literal[
    "revenue",
    "operating_expense",
    "payroll",
    "utilities",
    "rent",
    "taxes",
    "insurance",
    "interest_expense",
    "financing_proceeds",
    "capex",
    "financing",
    "unrestricted_subsidiary_transfer",
    "other",
]

# Direct operating cost of the revenue-generating activity — NOT corporate overhead.
# Used for Group-capex/EBITDA; related-party /opex clauses still read operating_expense
# (with expense-like fallback when empty). Do not expand without a document basis.
EBITDA_OPEX_CATEGORIES: tuple[str, ...] = ("operating_expense",)


class CarveOut(BaseModel):
    """Exclusion from covenant aggregation stated in the clause text itself."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["counterparty", "category", "txn_id", "description_keyword", "amount_below"]
    value: str  # counterparty / category / TXN-id / keyword
    amount: float | None = None  # only for amount_below (de-minimis threshold)
    quote: str  # verbatim carve-out wording from the clause


class ExprNode(BaseModel):
    """Flat expression-tree node (integer child refs — no recursive JSON schema)."""

    model_config = ConfigDict(extra="forbid")

    id: int
    op: Literal[
        "const",
        "agg",
        "related_outflows",
        "group_capex",
        "off_ledger",
        "add",
        "sub",
        "div",
        "max",
        "min",
    ]
    value: float | None = None  # const
    categories: list[CategoryName] | None = None  # agg / off_ledger
    sign: Literal["abs", "signed", "outflows", "inflows"] = "abs"  # agg
    aggregation: Literal["period_sum", "as_of_balance"] = "period_sum"
    children: list[int] = []  # add/sub/div/max/min


class CovenantSpec(BaseModel):
    """One financial-covenant clause, extracted from the active loan agreement.

    Open-set families plus generic sum/ratio/max fallbacks for private-set types.
    Escape hatch `unknown` + formula_description for shapes that fit none of the above.
    """

    model_config = ConfigDict(extra="forbid")

    clause: str  # e.g. "6.1"
    covenant_type: Literal[
        "ratio_group_capex_to_ebitda",
        "sum_revenue",
        "sum_related_party_payments",
        "sum_category",
        "ratio_generic",
        "max_category",
        "revenue_minus_max_category",
        "unknown",
    ]
    comparator: Literal["max", "min"]  # max: actual must not exceed threshold; min: must be >= threshold
    threshold: float
    period_start: str  # YYYY-MM-DD, from the clause text (not a default calendar year)
    period_end: str
    quote: str  # verbatim clause text this spec was extracted from
    # Free-text measurement description when covenant_type is "unknown"; else "".
    formula_description: str = ""
    # Singular category fields: Python/tests only. Excluded from LLM JSON schema —
    # Anthropic caps nullable unions at 16; singulars duplicated the list forms.
    filter_category: SkipJsonSchema[CategoryName | None] = None
    numerator_category: SkipJsonSchema[CategoryName | None] = None
    denominator_category: SkipJsonSchema[CategoryName | None] = None
    # Multi-category form (LLM fills these; empty list = not applicable).
    filter_categories: list[CategoryName] = []
    numerator_categories: list[CategoryName] = []
    denominator_categories: list[CategoryName] = []
    # Springing ratio: evaluate only when |numerator| exceeds this; else COMPLIANT with real actual.
    springing_min_numerator: float | None = None
    # period_sum = accumulate over [start,end]; as_of_balance = cumulative stock through period_end.
    aggregation: Literal["period_sum", "as_of_balance"] = "period_sum"
    # Clause-local exclusions («за исключением …»); verified against document text before apply.
    carve_outs: list[CarveOut] = []
    # Expression tree: Python compile_from_enum / tests / --engine tree only.
    # SkipJsonSchema: nesting + 4 nullable unions blew Anthropic's "Schema is too complex".
    expr_nodes: SkipJsonSchema[list[ExprNode] | None] = None
    expr_root: SkipJsonSchema[int | None] = None


class LoanAgreementFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    borrower_name: str
    account_id: str | None  # ledger account id as written (e.g. ACC-… / TELE-…), or null if absent
    contract_date: str  # YYYY-MM-DD, the agreement's own dated preamble
    effective_from: str  # YYYY-MM-DD, covenant-period start per the preamble
    effective_to: str  # YYYY-MM-DD, covenant-period end per the preamble
    is_superseded: bool  # true if the doc states it was replaced/superseded (any stamp or text)
    covenants: list[CovenantSpec]
    # EUR→USD disclosure in the same document (was a separate FxRateFacts LLM call).
    eur_usd_rate: float | None = None
    eur_amount: float | None = None
    usd_amount: float | None = None
    fx_quote: str = ""


class RelatedParty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counterparty_name: str
    voting_rights_pct: float
    quote: str
    # Share of subsidiary assets under security/pledge («в залоге»). Null if not disclosed.
    pledged_assets_pct: float | None = None


class KYCFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    borrower_name: str
    account_id: str | None  # ledger account id as written, or null when dossier has no account anchor
    related_party_threshold_pct: float | None  # ">= X%" rule; null if the dossier does not state one
    counterparties: list[RelatedParty]  # all disclosed, not just the ones over threshold
    # Below this pledge % the dossier treats the sub as Unrestricted («Неограниченн»). Null = not stated.
    unrestricted_pledge_threshold_pct: float | None = None


class Reclassification(BaseModel):
    """One auditor adjustment applied by S4 before covenant evaluation."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["reclassify", "exclude_from_period", "include_in_period"]
    # applied = auditor put the adjustment into the figures; other values are skipped by S4.
    status: Literal["applied", "considered_but_rejected", "referred_elsewhere", "informational"] = "applied"
    txn_id: str | None = None
    counterparty: str | None = None
    amount: float | None = None  # absolute magnitude as stated in the note, if any
    from_category: CategoryName | None = None
    to_category: CategoryName | None = None
    quote: str


class OffLedgerAddition(BaseModel):
    """Amount disclosed for covenant aggregation but absent from the ledger export."""

    model_config = ConfigDict(extra="forbid")

    amount: float  # signed as economics: expenses negative, or absolute with category implying expense
    category: CategoryName | None = None
    quote: str
    # When the note names a TXN id, this is ledger amount-fill — not a true off-ledger add.
    # S4 skips non-null target_txn_id so restored ledger rows are not double-counted.
    target_txn_id: str | None = None


class AuditorNotesFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    borrower_name: str
    reclassifications_present: bool
    reclassification_notes: list[str]  # verbatim quotes; empty if reclassifications_present is false
    reclassifications: list[Reclassification] = []  # structured actions for S4
    # Disclosures like severance obligations that must enter covenant math but have no TXN row.
    off_ledger_additions: list[OffLedgerAddition] = []
    # Materiality floor for off-ledger add-backs only (not reclassifications). Null = keep all.
    materiality_threshold_usd: float | None = None
    # EUR→USD disclosure in the same document (was a separate FxRateFacts LLM call).
    eur_usd_rate: float | None = None
    eur_amount: float | None = None
    usd_amount: float | None = None
    fx_quote: str = ""


class FxRateFacts(BaseModel):
    """EUR→USD rate disclosed in a borrower document (never an external lookup).

    Prefer a stated numeric rate in eur_usd_rate. When the document only gives a
    pair of EUR/USD amounts, fill eur_amount / usd_amount and leave eur_usd_rate
    null — S4 divides (LLM must not do arithmetic).
    """

    model_config = ConfigDict(extra="forbid")

    eur_usd_rate: float | None  # None when the document does not state a numeric rate
    eur_amount: float | None = None  # absolute EUR side of an implied-rate pair
    usd_amount: float | None = None  # absolute USD side of an implied-rate pair
    quote: str  # verbatim supporting text, or "" when nothing is found


class GroupCapexFacts(BaseModel):
    """Extracted from the ultimate parent's consolidated statements.

    LLM extracts components only; S4 computes the roll-forward
    (closing − opening + depreciation + disposals) and cross-checks against
    a direct cash-flow PP&E line when present.
    """

    model_config = ConfigDict(extra="forbid")

    # Extract-only for LLM/audit; S4 compute_group_capex does not read this field.
    parent_company_name: str
    opening_nbv: float | None = None
    closing_nbv: float | None = None
    depreciation: float | None = None
    disposals: float | None = None  # carrying value of disposals; 0 if note says none
    direct_capex_line: float | None = None  # CFS "purchases of PP&E" if present
    capex_value: float | None = None  # legacy / precomputed; S4 prefers components
    derivation: str  # what the note shows (no arithmetic required)
    quote: str


class TransactionCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txn_id: str
    category: CategoryName
    # R1: False = filler/noise for covenant aggregates (LLM + heuristics). Default True
    # so older cache entries without the field still validate via model defaults… —
    # field is required with default for forward-compatible schema.
    is_covenant_relevant: bool = True


class TransactionCategoryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    categories: list[TransactionCategory]


class RelatedPartyOverlapConfirm(BaseModel):
    """LLM confirmation that a token-overlap ledger counterparty is a disclosed related party."""

    model_config = ConfigDict(extra="forbid")

    counterparty_name: str
    is_related: bool
    voting_rights_pct: float | None = None
    quote: str  # verbatim supporting text, or "" when not related / not found


class RelatedPartyOverlapBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmations: list[RelatedPartyOverlapConfirm]


class DocTypeGuess(BaseModel):
    """LLM fallback when regex classification finds no loan_agreement (A4)."""

    model_config = ConfigDict(extra="forbid")

    doc_type: Literal[
        "loan_agreement",
        "kyc_dossier",
        "auditor_notes",
        "consolidated_parent_report",
        "other",
    ]
    quote: str  # verbatim supporting snippet; required for non-other
