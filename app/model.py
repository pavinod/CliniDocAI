"""Configurable model adapter for claim-support assessment.

Every clinical claim the pipeline surfaces carries evidence quotes. A valid
quote only proves provenance — this module decides separately whether the
quote actually supports the claim, and unsupported claims are dropped.

The default adapter is deterministic (no network, no secrets): it applies a
keyword-overlap heuristic and can be replaced by an HTTP LLM adapter via
CLINIDOC_MODEL=openai-compatible + CLINIDOC_MODEL_URL/CLINIDOC_MODEL_KEY.
Timeouts and invalid output raise ModelError, which the screen layer turns
into a recoverable error state rather than a crash.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Protocol


class ModelError(Exception):
    """Recoverable model failure (timeout, invalid output)."""


class ModelAdapter(Protocol):
    name: str

    def assess_support(self, claim: str, quote: str,
                       original: str | None = None) -> str:
        """Return 'supported' | 'unsupported' | 'unchecked'."""
        ...


_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is",
         "was", "are", "be", "as", "by", "for", "with", "not", "no", "der",
         "die", "das", "und", "im", "am", "bei", "mit", "auf", "den", "dem",
         "ein", "eine", "einer", "eines", "ist", "sich", "nicht"}


_NUMERIC_UNIT = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>"
    r"kg\s*/\s*m(?:\s*2|²)|cm\s*/\s*(?:sek|s)|mmol\s*/\s*l|mg\s*/\s*dl|"
    r"years?|months?|days?|mm\s*hg|°c|mcg|µg|μg|ug|/min|mg|ml|kg|cm|mm|g|l|"
    r"%(?:igen|iger|igem|iges|ige)?)"
    r"(?![a-z])",
    re.IGNORECASE,
)


def _canonical_number(value: str) -> str:
    number = format(Decimal(value.replace(",", ".")).normalize(), "f")
    return number.rstrip("0").rstrip(".") if "." in number else number


def _canonical_unit(unit: str) -> str:
    normalized = re.sub(r"\s+", "", unit.casefold()).replace("μ", "µ")
    if normalized.startswith("%"):
        return "%"
    if normalized in {"µg", "mcg", "ug"}:
        return "ug"
    if normalized == "cm/sek":
        return "cm/s"
    if normalized in {"kg/m²", "kg/m2"}:
        return "kg/m2"
    return normalized


def _numeric_mentions(text: str) -> set[tuple[str, str]]:
    return {
        (_canonical_number(match.group("value")),
         _canonical_unit(match.group("unit")))
        for match in _NUMERIC_UNIT.finditer(text)
    }


def _toks(s: str) -> set[str]:
    normalized = _NUMERIC_UNIT.sub(
        lambda match: f"{_canonical_number(match.group('value'))} "
        f"{_canonical_unit(match.group('unit'))}",
        s.casefold(),
    )
    return {t for t in re.findall(r"[a-zäöüß0-9.%/]+", normalized)
            if t not in _STOP}


def _normalized_phrase(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def support_mismatch_reason(claim: str, quote: str) -> str | None:
    ct, qt = _toks(claim), _toks(quote)
    if "left" in ct or "links" in ct:
        ct.add("left")
    if "right" in ct or "rechts" in ct:
        ct.add("right")
    if "left" in qt or "links" in qt:
        qt.add("left")
    if "right" in qt or "rechts" in qt:
        qt.add("right")
    claim_left = bool(ct & {"left", "links"})
    claim_right = bool(ct & {"right", "rechts"})
    quote_left = bool(qt & {"left", "links"})
    quote_right = bool(qt & {"right", "rechts"})
    if (claim_left and quote_right and not quote_left) or (
            claim_right and quote_left and not quote_right):
        return "laterality mismatch: claim and quote specify opposite sides"

    quote_numbers = _numeric_mentions(quote)
    for number, unit in _numeric_mentions(claim):
        if (number, unit) not in quote_numbers:
            return (f"number/unit mismatch: expected {number} {unit} "
                    "is absent from quote")
    return None


class DeterministicAdapter:
    """Keyword-overlap heuristic — deterministic test double for CI.

    Marks a claim 'supported' when enough of the claim's content words appear
    in the quote (or the claim is derived/quoted verbatim), 'unsupported' when
    coverage is very low, else 'unchecked' (kept, labelled as not verified).
    Never invents text: it only judges existing claims.
    """

    name = "deterministic-v1"

    def unsupported_reason(self, claim: str, quote: str) -> str | None:
        return support_mismatch_reason(claim, quote)

    def assess_support(self, claim: str, quote: str,
                       original: str | None = None) -> str:
        if support_mismatch_reason(claim, quote):
            return "unsupported"

        if (original and _normalized_phrase(original)
                and _normalized_phrase(original) in _normalized_phrase(quote)):
            return "supported"
        ct, qt = _toks(claim), _toks(quote)
        if not ct:
            return "unchecked"
        if claim.strip().lower() in quote.lower():
            return "supported"
        overlap = len(ct & qt) / len(ct)
        if overlap >= 0.6:
            return "supported"
        return "unchecked"


class HttpAdapter:
    """Minimal OpenAI-compatible chat adapter. Optional — only when configured."""

    name = "http-v1"

    def __init__(self, url: str, key: str, timeout: float = 20.0,
                 model_name: str | None = None):
        self.url, self.key, self.timeout = url, key, timeout
        self.model_name = model_name or os.environ.get(
            "CLINIDOC_MODEL_NAME", "gpt-4o-mini")

    def assess_support(self, claim: str, quote: str,
                       original: str | None = None) -> str:
        prompt = (
            "Decide whether QUOTE supports CLAIM. Answer exactly one token: "
            "supported, unsupported, or unchecked.\n"
            f"QUOTE: {quote}\nORIGINAL WORDING: {original or ''}\nCLAIM: {claim}")
        text = self.complete(prompt, max_tokens=4).strip().lower()
        if text not in ("supported", "unsupported", "unchecked"):
            raise ModelError(f"invalid model verdict: {text!r}")
        return text

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """One chat completion; raises ModelError on any transport problem."""
        body = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ModelError(f"model request failed: {e}") from e
        except TimeoutError as e:
            raise ModelError("model request timed out") from e
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError("invalid model output") from e


# Available OpenRouter models (shown in the UI selector)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("openrouter/free",                              "OpenRouter — free tier"),
    ("deepseek/deepseek-v4-flash",                   "DeepSeek v4 Flash"),
    ("deepseek/deepseek-v4.1-flash-1017",            "DeepSeek v4.1 Flash"),
    ("z-ai/glm-4.7-flash",                           "GLM-4.7 Flash"),
    ("z-ai/glm-flash-latest",                        "GLM Flash Latest"),
    ("moonshotai/kimi-latest",                       "Kimi Latest"),
    ("anthropic/claude-haiku-5.5",                   "Claude Haiku 5.5"),
    ("openai/gpt-5-nano",                            "GPT-5 Nano"),
    ("openai/gpt-5.6-luna",                          "GPT-5.6 Luna"),
    ("openai/gpt-5.1-codex-mini",                    "GPT-5.1 Codex Mini"),
    ("google/gemini-3.1-flash-lite",                 "Gemini 3.1 Flash Lite"),
    ("qwen/qwen3-coder-30b-a3b-instruct",            "Qwen3 Coder 30B"),
    ("qwen/qwen3-vl-30b-a3b-instruct",               "Qwen3 VL 30B"),
    ("mistralai/mistral-small-3.2-24b-instruct",     "Mistral Small 3.2 24B"),
    ("meta-llama/llama-4-scout",                     "Llama 4 Scout"),
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RULE_BASED_CHOICE = "none"
LMSTUDIO_CHOICE = "lmstudio"


def model_choices() -> list[tuple[str, str]]:
    """(choice id, label) pairs for the UI selector; rule-based first."""
    return ([(RULE_BASED_CHOICE, "Rule-based only (no AI model)")]
            + [(f"openrouter:{model_id}", f"OpenRouter · {label}")
               for model_id, label in OPENROUTER_MODELS]
            + [(LMSTUDIO_CHOICE, "LM Studio (local model)")])


def adapter_for_choice(choice: str) -> tuple["ModelAdapter | None", str | None]:
    """Build the adapter for a UI choice without touching os.environ.

    Returns (adapter, problem). A missing API key yields (None, message) so the
    caller can stay rule-based and tell the user why.
    """
    timeout = float(os.environ.get("MODEL_TIMEOUT_S",
                                   os.environ.get("CLINIDOC_MODEL_TIMEOUT", "8")))
    if choice.startswith("openrouter:"):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return None, "OPENROUTER_API_KEY is not set"
        return HttpAdapter(f"{OPENROUTER_BASE_URL}/chat/completions", key,
                           timeout, choice.removeprefix("openrouter:")), None
    if choice == LMSTUDIO_CHOICE:
        base = os.environ.get("LM_STUDIO_URL",
                              "http://localhost:1234/v1").rstrip("/")
        return HttpAdapter(f"{base}/chat/completions",
                           os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
                           timeout, os.environ.get("LM_STUDIO_MODEL", "local")), None
    return DeterministicAdapter(), None


def get_adapter() -> ModelAdapter:
    """Adapter from env config; deterministic by default (safe for CI).

    Supports the scaffold variable names:
      MODEL_PROVIDER=none            -> deterministic rules only
      MODEL_PROVIDER=openrouter      -> OPENROUTER_API_KEY / OPENROUTER_MODEL / MODEL_TIMEOUT_S
      MODEL_PROVIDER=openai          -> OPENAI_API_KEY / OPENAI_MODEL / MODEL_TIMEOUT_S
      MODEL_PROVIDER=lmstudio        -> LM_STUDIO_URL / LM_STUDIO_API_KEY / LM_STUDIO_MODEL
    Legacy override: CLINIDOC_MODEL=http + CLINIDOC_MODEL_URL.
    """
    provider = os.environ.get("MODEL_PROVIDER", os.environ.get("CLINIDOC_MODEL", "none"))
    timeout = float(os.environ.get("MODEL_TIMEOUT_S",
                                   os.environ.get("CLINIDOC_MODEL_TIMEOUT", "8")))
    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
        return HttpAdapter(
            f"{OPENROUTER_BASE_URL}/chat/completions", key, timeout, model)
    if provider == "openai":
        return HttpAdapter("https://api.openai.com/v1/chat/completions",
                           os.environ.get("OPENAI_API_KEY", ""), timeout,
                           os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "lmstudio":
        base = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1").rstrip("/")
        return HttpAdapter(f"{base}/chat/completions",
                           os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
                           timeout, os.environ.get("LM_STUDIO_MODEL", "local"))
    if provider == "http":
        url = os.environ.get("CLINIDOC_MODEL_URL")
        if not url:
            raise ModelError("CLINIDOC_MODEL_URL not set for http adapter")
        return HttpAdapter(url, os.environ.get("CLINIDOC_MODEL_KEY", ""), timeout)
    return DeterministicAdapter()
