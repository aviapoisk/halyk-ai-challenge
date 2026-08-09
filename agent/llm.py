"""S1-S6 shared LLM wrapper — the only place in the codebase that calls the model.

docs/PLAN_PROMPT.md §4 asks for "temperature 0" determinism; that parameter is
removed on current-generation Anthropic models (400 error) and never guaranteed
identical output anyway. Determinism here comes entirely from the sidecar response
cache (content-addressed by model + effort + thinking + system + user content +
schema), matching docs/STRATEGY.md §6's committed-cache design and the organizers'
rerun requirement (docs/PLAN_PROMPT.md §3.2) — a warm cache reproduces byte-identical
answers regardless of model sampling behavior.

Providers (AGENT_PROVIDER):
- anthropic — AsyncAnthropic; adaptive thinking + output_config.effort/json_schema
- openai — AsyncOpenAI (LM Studio / vLLM / …); temperature 0 + response_format json_schema

Adaptive thinking (`thinking.type=adaptive`) is on by default for Anthropic and
toggled via `AGENT_ADAPTIVE_THINKING`; depth is steered by `output_config.effort`
(`AGENT_EFFORT`). Structured output uses json_schema rather than tool calls: every
extraction schema in schemas.py must set `model_config = ConfigDict(extra="forbid")`
so its generated JSON schema carries `additionalProperties: false`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any, TypeVar

import anthropic
import openai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from agent.config import Config
from agent.progress import NullSink, ProgressSink

T = TypeVar("T", bound=BaseModel)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    # APITimeoutError deliberately NOT retried: local multimodal hangs would otherwise
    # burn stop_after_attempt × timeout per page (vision guard needs a fast fail).
)

# LM Studio reports a crashed/unloaded local model as HTTP 400 BadRequestError
# ("The model has crashed…" / "Model is not loaded"). The next request triggers a
# JIT reload, so these are transient and must be retried like a 5xx — in the
# 2026-08-08 run one crash instantly failed 7 of 12 scenarios with zero retries.
_RETRYABLE_400_MARKERS = ("model has crashed", "model is not loaded", "no model loaded")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, openai.BadRequestError):
        message = str(exc).lower()
        return any(marker in message for marker in _RETRYABLE_400_MARKERS)
    return False


def _require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Anthropic structured-output: optional fields explode grammar size; mark all required.

    Pydantic fields with defaults are omitted from ``required``; for LLM output we still
    want every key present. Mutates a deep copy.
    """
    out = json.loads(json.dumps(schema))

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        props = node.get("properties")
        if isinstance(props, dict) and props:
            node["required"] = sorted(props.keys())
            for child in props.values():
                walk(child)
        for key in ("$defs", "definitions"):
            defs = node.get(key)
            if isinstance(defs, dict):
                for child in defs.values():
                    walk(child)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items)
        for key in ("anyOf", "oneOf", "allOf"):
            alts = node.get(key)
            if isinstance(alts, list):
                for alt in alts:
                    walk(alt)

    walk(out)
    return out


def _cache_key(
    model: str,
    effort: str,
    thinking: str,
    system: str,
    user_content: str,
    schema_name: str,
    schema_json: str,
    image_cache_token: str | None = None,
) -> str:
    """Content-addressed cache key. Text-only calls omit image_cache_token so keys stay stable."""
    h = hashlib.sha256()
    parts = [model, effort, thinking, system, user_content, schema_name, schema_json]
    if image_cache_token is not None:
        parts.append(image_cache_token)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class LLMClient:
    """The single client wrapper for the whole pipeline (S2-S6)."""

    def __init__(self, config: Config, sink: ProgressSink | None = None):
        self.config = config
        self.sink: ProgressSink = sink or NullSink()
        self._provider = (config.model_provider or "anthropic").strip().lower()
        self._semaphore = asyncio.Semaphore(config.llm_concurrency)

        if self._provider == "openai":
            kwargs: dict[str, Any] = {"api_key": config.api_key or "lm-studio"}
            if config.base_url:
                kwargs["base_url"] = config.base_url
            self._openai = openai.AsyncOpenAI(**kwargs)
            self._anthropic = None
        elif self._provider == "anthropic":
            self._anthropic = anthropic.AsyncAnthropic(api_key=config.api_key)
            self._openai = None
        else:
            raise ValueError(f"unsupported AGENT_PROVIDER={config.model_provider!r}; use anthropic|openai")

    def _thinking_param(self) -> dict[str, str]:
        if self.config.llm_adaptive_thinking:
            return {"type": "adaptive"}
        return {"type": "disabled"}

    @staticmethod
    def _user_content_anthropic(user_content: str, images: list[bytes] | None) -> str | list[dict[str, Any]]:
        if not images:
            return user_content
        blocks: list[dict[str, Any]] = []
        for png in images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(png).decode("ascii"),
                    },
                }
            )
        blocks.append({"type": "text", "text": user_content})
        return blocks

    @staticmethod
    def _user_content_openai(user_content: str, images: list[bytes] | None) -> str | list[dict[str, Any]]:
        if not images:
            return user_content
        blocks: list[dict[str, Any]] = []
        for png in images:
            b64 = base64.standard_b64encode(png).decode("ascii")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        blocks.append({"type": "text", "text": user_content})
        return blocks

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(5),
    )
    async def _call(
        self,
        system: str,
        user_content: str,
        schema_json: dict,
        schema_name: str,
        *,
        label: str = "",
        images: list[bytes] | None = None,
    ) -> tuple[str, int, int]:
        self.sink.llm_wait(schema_name, label=label)
        t0 = time.monotonic()
        async with self._semaphore:
            if self._provider == "openai":
                text, input_tokens, output_tokens = await self._call_openai(
                    system, user_content, schema_json, schema_name, images=images
                )
            else:
                text, input_tokens, output_tokens = await self._call_anthropic(
                    system, user_content, schema_json, schema_name, images=images
                )
        self.sink.llm_done(
            schema_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
            duration_s=time.monotonic() - t0,
            label=label,
        )
        return text, input_tokens, output_tokens

    async def _call_anthropic(
        self,
        system: str,
        user_content: str,
        schema_json: dict,
        schema_name: str,
        *,
        images: list[bytes] | None = None,
    ) -> tuple[str, int, int]:
        assert self._anthropic is not None
        response = await self._anthropic.messages.create(
            model=self.config.model_name,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": self._user_content_anthropic(user_content, images)}],
            thinking=self._thinking_param(),
            output_config={
                "effort": self.config.llm_effort,
                "format": {"type": "json_schema", "schema": schema_json},
            },
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"model hit max_tokens before finishing structured output ({schema_name})")
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
        if not text_blocks:
            raise RuntimeError(f"model response has no text block (stop_reason={response.stop_reason!r}, schema={schema_name})")
        usage = response.usage
        return (
            text_blocks[0],
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    async def _call_openai(
        self,
        system: str,
        user_content: str,
        schema_json: dict,
        schema_name: str,
        *,
        images: list[bytes] | None = None,
    ) -> tuple[str, int, int]:
        assert self._openai is not None
        # Local OpenAI-compatible servers often hang on unsupported multimodal inputs;
        # bound vision calls so ocr_scan_document can flag vision_failed instead of stalling.
        create_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": 0,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content_openai(user_content, images)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_json,
                    "strict": True,
                },
            },
        }
        if images:
            create_kwargs["timeout"] = 60.0
        response = await self._openai.chat.completions.create(**create_kwargs)
        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise RuntimeError(f"model response has no choices (schema={schema_name})")
        if choice.finish_reason == "content_filter":
            raise RuntimeError(f"model refused via content_filter (schema={schema_name})")
        if choice.finish_reason == "length":
            raise RuntimeError(f"model hit max_tokens before finishing structured output ({schema_name})")
        text = (choice.message.content or "").strip() if choice.message else ""
        if not text:
            raise RuntimeError(f"model response has no text (finish_reason={choice.finish_reason!r}, schema={schema_name})")
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return text, input_tokens, output_tokens

    async def extract(
        self,
        system: str,
        user_content: str,
        schema: type[T],
        cache_dir: Path | None = None,
        use_cache: bool = True,
        *,
        label: str = "",
        images: list[bytes] | None = None,
        image_cache_token: str | None = None,
    ) -> T:
        """Runs one structured-output call, validated against `schema`, cached by content hash.

        Optional `images` are PNG page renders for vision OCR. Cache keys for vision
        use `image_cache_token` (sha256 + page + render params), never raw PNG bytes.
        """
        schema_json = _require_all_properties(schema.model_json_schema())
        schema_name = schema.__name__
        thinking = "adaptive" if self.config.llm_adaptive_thinking else "disabled"
        key = _cache_key(
            self.config.model_name,
            self.config.llm_effort,
            thinking,
            system,
            user_content,
            schema_name,
            json.dumps(schema_json, sort_keys=True),
            image_cache_token=image_cache_token,
        )
        cache_file = (cache_dir / "llm" / f"{key}.json") if cache_dir else None

        if use_cache and cache_file and cache_file.exists():
            self.sink.llm_done(schema_name, cache_hit=True, label=label)
            return schema.model_validate_json(cache_file.read_text(encoding="utf-8"))

        raw, _in_tok, _out_tok = await self._call(
            system, user_content, schema_json, schema_name, label=label, images=images
        )
        result = schema.model_validate_json(raw)

        if use_cache and cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(result.model_dump_json(), encoding="utf-8")

        return result
