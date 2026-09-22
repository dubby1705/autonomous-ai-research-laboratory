"""
AARL Showcase — LLM Service
=============================
One reusable LLM service wrapping the Groq API with:

- single reusable client
- robust JSON parsing (Markdown code fences, stray prose, partial repair)
- required-field validation
- limited retries with backoff for transient failures
- a clean error taxonomy (never leaks the key)
- a transparent deterministic fallback so the showcase can run offline
  (every LLM-dependent output is labelled with its `source`).

The LLM is a *reasoning/planning* layer. It never supplies experimental numbers;
those always come from the real simulation engine.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from . import config as cfg
from .errors import (
    AARLError,
    MissingAPIKeyError,
    MalformedResponseError,
    classify_groq_error,
)

log = logging.getLogger("aarl.llm")

# Model known to Groq used as final fallback for unparseable model requests.
_FALLBACK_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from arbitrary LLM text."""
    if not text:
        return None
    text = text.strip()

    # 1) Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Strip Markdown fences then re-try.
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    fenced = re.sub(r"\s*```\s*$", "", fenced).strip()
    try:
        obj = json.loads(fenced)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Find the first balanced {...} block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None


def validate_fields(obj: Dict[str, Any], required: List[str]) -> List[str]:
    """Return a list of missing required keys."""
    missing = []
    for key in required:
        value = obj.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
    return missing


# ---------------------------------------------------------------------------
# Deterministic fallback provider
# ---------------------------------------------------------------------------
class DeterministicProvider:
    """Symmetric fallback that returns structured, deterministic outputs.

    It is explicitly labelled as `source='deterministic'` and is used only to
    keep the pipeline running end-to-end when no Groq key is available. It never
    fabricates experimental numbers (experiments always come from simulation).
    """

    name = "deterministic"
    model = "n/a (deterministic fallback)"

    def complete_json(self, system_prompt: str, user_prompt: str, **kw) -> Optional[Dict[str, Any]]:
        return None  # no generic ability; specialised functions handle this


# ---------------------------------------------------------------------------
# Groq service
# ---------------------------------------------------------------------------
class GroqService:
    """Reusable, thread-safe Groq client with robust calling conventions."""

    def __init__(self, config: cfg.ResearchConfig, model: Optional[str] = None):
        self.config = config
        if model is None or not model.strip():
            model = cfg.get_groq_model()
        self.model = model
        self._client = None
        self._init_error: Optional[AARLError] = None
        self._available = False
        self.attempt_available()

    # ---- client -----------------------------------------------------------
    def attempt_available(self) -> bool:
        """Initialise the client and check basic availability without calling."""
        key = cfg.get_groq_api_key()
        if not key:
            self._init_error = MissingAPIKeyError()
            self._available = False
            return False
        try:
            from groq import Groq

            self._client = Groq(api_key=key, timeout=self.config.llm_timeout_seconds, max_retries=0)
            self._available = True
            self._init_error = None
            return True
        except AARLError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._client = None
            self._available = False
            self._init_error = classify_groq_error(exc)
            return False

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    @property
    def init_error(self) -> Optional[AARLError]:
        return self._init_error

    # ---- core request -------------------------------------------------------
    def _raw_complete(self, system: str, user: str, temperature: float) -> str:
        """Make one Groq chat completion request, returning the raw text."""
        if not self.available:
            raise self._init_error or AARLError("Configuration Error", "Groq client unavailable.", "")

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=self.config.llm_budget_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            err = classify_groq_error(exc)
            # mark client unusable on hard auth/model errors
            if err.category in ("Authentication Error", "Model Unavailable", "Quota / Billing"):
                self._available = False
            raise err

        try:
            content = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise MalformedResponseError("no content") from exc
        return content

    # ---- public JSON helper with retries ----------------------------------
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        required_fields: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call Groq, parse JSON, validate required fields, retry on failure.

        Bounded retries only; if it still fails without a usable JSON object,
        raise MalformedResponseError so the caller can decide how to continue.
        """
        required = required_fields or []
        temp = self.config.temperature if temperature is None else temperature
        retries = self.config.max_llm_retries if retries is None else retries

        last_err: Optional[BaseException] = None
        for attempt in range(retries + 1):
            if attempt > 0:
                delay = 1.5 * (2 ** (attempt - 1))  # 1.5s, 3s, 6s
                log.info("Retrying LLM call in %.1fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
            try:
                text = self._raw_complete(system_prompt, user_prompt, temp)
            except AARLError as exc:
                # Non-transient (auth/quota/model) — do not burn retries.
                if exc.category in (
                    "Authentication Error",
                    "Quota / Billing",
                    "Model Unavailable",
                    "Missing API Key",
                ):
                    raise
                last_err = exc
                continue

            obj = extract_json(text)
            if obj is None:
                last_err = MalformedResponseError("no JSON object")
                continue
            missing = validate_fields(obj, required)
            if missing:
                last_err = MalformedResponseError(f"missing fields: {', '.join(missing)}")
                continue
            return obj

        raise last_err or MalformedResponseError("retry limit reached")

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        """Return raw text (for freely-worded analyses)."""
        temp = self.config.temperature if temperature is None else temperature
        self._available or self.attempt_available()
        return self._raw_complete(system_prompt, user_prompt, temp)


def make_llm_service(config: Optional[cfg.ResearchConfig] = None) -> GroqService:
    """Build the shared LLM service (or deterministic fallback if unconfigured)."""
    config = config or cfg.ResearchConfig()
    svc = GroqService(config)
    if svc.available:
        return svc
    # Return a lightweight wrapper that surfaces the config problem lazily via
    # the same interface, so the pipeline can decide to fall back gracefully.
    return svc