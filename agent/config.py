"""Runtime config: env + CLI flags. No provider/model names or dataset paths hardcoded here.

docs/PLAN_PROMPT.md §6: model name and provider come from the environment
(`env.example` carries placeholders); the code never names a specific model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_PROFILE_PRESETS: dict[str, dict] = {
    "default": {},
    "fast": {"llm_effort": "medium", "llm_concurrency": 16},
    "careful": {"llm_effort": "xhigh", "llm_concurrency": 8},
}


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    out_path: Path
    cache_dir: Path
    profile: str = "default"
    seed: int = 42
    use_cache: bool = True
    only_scenario: str | None = None
    # Dev-only: agent.run --dev writes logs/dev_dump.json (per-scenario documents,
    # extracts, transactions) for tools/dev.py. Never used on the battle path.
    dev: bool = False
    # enum | shadow | tree — submission uses enum unless engine=tree (manual gate).
    engine: str = "shadow"

    # S1: a page is routed to the vision fallback when its extracted text is
    # shorter than this many characters (docs/FINDINGS-2026-08-06.md §1).
    vision_char_threshold: int = 50

    # Model/provider come from the environment, never hardcoded in code paths that
    # touch the dataset (docs/PLAN_PROMPT.md §6). "claude-opus-5" is only the
    # fallback when AGENT_MODEL is unset.
    model_name: str = os.environ.get("AGENT_MODEL", "") or "claude-opus-5"
    model_provider: str = os.environ.get("AGENT_PROVIDER", "anthropic")
    # OpenAI-compatible base URL (LM Studio default: http://localhost:1234/v1)
    base_url: str = os.environ.get("AGENT_BASE_URL", "")
    # output_config.effort: low | medium | high | xhigh | max
    llm_effort: str = os.environ.get("AGENT_EFFORT", "") or "high"
    # thinking.type: adaptive (default on Claude 5) | disabled
    llm_adaptive_thinking: bool = _env_bool("AGENT_ADAPTIVE_THINKING", True)
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    team: str = os.environ.get("TEAM_NAME", "")
    contact_email: str = os.environ.get("CONTACT_EMAIL", "")

    llm_concurrency: int = int(os.environ.get("AGENT_LLM_CONCURRENCY", "12"))
    self_consistency: bool = _env_bool("AGENT_SELF_CONSISTENCY", True)


def load_config(
    dataset_dir: Path,
    out_path: Path,
    cache_dir: Path,
    profile: str = "default",
    seed: int = 42,
    use_cache: bool = True,
    only_scenario: str | None = None,
    dev: bool = False,
    engine: str = "shadow",
) -> Config:
    cfg = Config(
        dataset_dir=dataset_dir,
        out_path=out_path,
        cache_dir=cache_dir,
        profile=profile,
        seed=seed,
        use_cache=use_cache,
        only_scenario=only_scenario,
        dev=dev,
        engine=engine,
    )
    preset = _PROFILE_PRESETS.get(profile, {})
    if preset:
        cfg = replace(cfg, **preset)
    return cfg
