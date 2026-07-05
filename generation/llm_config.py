"""
LLM provider abstraction for RedSea GPT.

The system supports two wire protocols so we can talk to very different hosted
LLMs through one clean factory:

* ``anthropic_messages`` - the **Anthropic Messages API** shape
  (``POST /v1/messages``, header ``x-api-key``, body ``{model, messages,
  max_tokens}``, response ``content[].text``). This is what **OptoLLM**
  (``https://optollm.optomatica.com``) speaks.
* ``openai_chat`` - the classic **OpenAI Chat Completions** shape
  (``POST /chat/completions``, header ``Authorization: Bearer``, response
  ``choices[0].message.content``). This is what **Groq**, **OpenAI**, and most
  OpenAI-compatible proxies speak.

Provider presets
----------------
+--------------------+------------------+--------------------------+
| provider           | protocol         | env key                  |
+--------------------+------------------+--------------------------+
| ``optillm``        | anthropic_messages| ``OPTO_LLM_API_KEY``     |
| ``groq``           | openai_chat      | ``GROQ_API_KEY``         |
| ``openai``         | openai_chat      | ``OPENAI_API_KEY``       |
| ``openai-compatible``| openai_chat    | ``LLM_API_KEY``          |
+--------------------+------------------+--------------------------+

Selection order: explicit arg -> ``LLM_PROVIDER`` env -> auto-detect from which
``*_API_KEY`` is present (optillm first).

Credentials are stored as ``SecretStr`` so they never appear in ``repr()``,
logs, tracebacks, or error messages. This module contains **no hardcoded keys**.

Engineering notes
-----------------
* OptoLLM sits behind Cloudflare, which returns HTTP 403 ``error 1010`` for
  Python's default ``User-Agent``. We therefore send a recognizable
  ``User-Agent`` on every request.
* Error responses are redacted (any ``sk-...`` fragment the proxy echoes back is
  scrubbed) before being surfaced, so logs/artifacts never leak key material.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun

# SecretStr masks the key in repr()/str()/logs across pydantic v1 and v2.
# langchain >=1.x ships pydantic v2; older versions exposed langchain_core.pydantic_v1.
try:
    from pydantic import Field, SecretStr  # pydantic v2 (langchain >=1.x)
except Exception:  # pragma: no cover
    try:
        from langchain_core.pydantic_v1 import Field, SecretStr  # pydantic v1 shim
    except Exception:  # pragma: no cover
        Field = None

        class SecretStr(str):  # type: ignore[no-redef]
            """Fallback: a str subclass that masks itself in repr."""
            def __repr__(self):
                return "**********"

# Best-effort .env load; never hard-fail if python-dotenv is absent.
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------
PROVIDER_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "optillm": {
        "protocol": "anthropic_messages",
        "env_key": "OPTO_LLM_API_KEY",
        "env_base": "OPTO_LLM_BASE_URL",
        "env_model": "OPTO_LLM_MODEL",
        "default_base": "https://optollm.optomatica.com/v1",
        "default_model": "mistral-small",
    },
    "groq": {
        "protocol": "openai_chat",
        "env_key": "GROQ_API_KEY",
        "env_base": "GROQ_BASE_URL",
        "env_model": "GROQ_MODEL",
        "default_base": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "protocol": "openai_chat",
        "env_key": "OPENAI_API_KEY",
        "env_base": "OPENAI_BASE_URL",
        "env_model": "OPENAI_MODEL",
        "default_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "openai-compatible": {
        "protocol": "openai_chat",
        "env_key": "LLM_API_KEY",
        "env_base": "LLM_BASE_URL",
        "env_model": "LLM_MODEL",
        "default_base": None,
        "default_model": None,
    },
}

VALID_PROVIDERS = tuple(PROVIDER_PRESETS.keys())

# Provider aliases. The product is branded "OptoLLM" (see the optollm.optomatica.com
# URL and the OPTO_LLM_* env vars), but the preset key was written as "optillm".
# Accept both spellings so a brand-correct ``LLM_PROVIDER=optollm`` resolves to the
# same configuration. Auto-detection via OPTO_LLM_API_KEY already works either way.
_PROVIDER_ALIASES = {"optollm": "optillm"}
VALID_PROTOCOLS = ("anthropic_messages", "openai_chat")
_USER_AGENT = "redsea-gpt/1.0 (+https://github.com/yaseen-elbeltagy)"
_KEY_FRAG_RE = re.compile(r"sk-[A-Za-z0-9_\-]{3,}")


def _redact(text: str) -> str:
    """Scrub any ``sk-...`` key fragment a server might echo back."""
    return _KEY_FRAG_RE.sub("sk-***REDACTED***", text or "")


def _detect_provider_from_env() -> str:
    for name in ("optillm", "groq", "openai"):
        if os.getenv(PROVIDER_PRESETS[name]["env_key"]):
            return name
    return ""


def resolve_provider_config(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve a fully-specified, secret-free-except-key provider config."""
    provider = (provider or os.getenv("LLM_PROVIDER") or _detect_provider_from_env()).strip().lower()
    # Resolve brand aliases ("optollm" -> "optillm") before validation, so the
    # canonical product name works as an LLM_PROVIDER value.
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if not provider:
        raise ValueError(
            "No LLM provider configured. Set one of OPTO_LLM_API_KEY / "
            "GROQ_API_KEY / OPENAI_API_KEY in your environment, or set "
            "LLM_PROVIDER explicitly. See .env.example."
        )
    if provider not in PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown provider '{provider}'. Valid providers: {VALID_PROVIDERS}."
        )

    preset = PROVIDER_PRESETS[provider]
    api_key = (api_key or os.getenv(preset["env_key"], "") or "").strip()
    base_url = (base_url or os.getenv(preset["env_base"], "") or preset["default_base"] or "").strip()
    model = (model or os.getenv(preset["env_model"], "") or preset["default_model"] or "").strip()

    if not api_key:
        raise ValueError(
            f"No API key for provider '{provider}'. Set {preset['env_key']} "
            "(see .env.example)."
        )
    if not base_url:
        raise ValueError(f"No base_url for provider '{provider}'. Set {preset['env_base']}.")
    if not model:
        raise ValueError(
            f"No model for provider '{provider}'. Set {preset['env_model']} "
            f"(e.g. OPTO_LLM_MODEL=mistral-small or gpt-4o-mini)."
        )

    return {
        "provider": provider,
        "protocol": preset["protocol"],
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
    }


def describe_active_provider() -> Dict[str, Any]:
    """Return a SECRET-FREE summary of the resolved provider (for CLI/logs)."""
    try:
        cfg = resolve_provider_config()
        return {
            "provider": cfg["provider"],
            "protocol": cfg["protocol"],
            "base_url": cfg["base_url"],
            "model": cfg["model"],
            "configured": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {"provider": None, "configured": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class UniversalLLM(BaseLLM):
    """One client, two wire protocols (Anthropic Messages + OpenAI Chat).

    The API key is held as a ``SecretStr`` and is never included in
    ``_identifying_params`` or error messages. All error bodies are redacted.
    """

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    base_url: str = ""
    model: str = ""
    provider: str = "openai-compatible"
    protocol: str = "openai_chat"
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 90.0

    @property
    def _llm_type(self) -> str:
        return self.provider

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        # Deliberately excludes the key.
        return {
            "provider": self.provider,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _key_value(self) -> str:
        """Return the raw key for HTTP use only; never appears in repr/logs."""
        if SecretStr is not None and isinstance(self.api_key, SecretStr):
            return self.api_key.get_secret_value()
        return str(self.api_key)

    # -- protocol helpers --------------------------------------------------
    def _build_request(self, prompt: str, stop: Optional[List[str]]) -> Dict[str, Any]:
        """Return (url, headers, payload) for the active protocol."""
        import requests  # noqa  (kept local so import errors are surfaced at call time)

        if self.protocol == "anthropic_messages":
            url = self.base_url.rstrip("/") + "/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._key_value(),
                "anthropic-version": "2023-06-01",
                # Cloudflare in front of OptoLLM blocks the default Python UA.
                "User-Agent": _USER_AGENT,
            }
            payload: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            return {"url": url, "headers": headers, "payload": payload, "lib": requests}

        # default: openai_chat
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key_value()}",
            "User-Agent": _USER_AGENT,
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop
        return {"url": url, "headers": headers, "payload": payload, "lib": requests}

    def _extract_text(self, data: Dict[str, Any]) -> str:
        if self.protocol == "anthropic_messages":
            blocks = data.get("content", []) or []
            return " ".join(
                b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        choices = data.get("choices") or []
        if choices:
            return (choices[0].get("message") or {}).get("content", "")
        return ""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        req = self._build_request(prompts[0] if prompts else "", stop)
        requests = req["lib"]
        generations: List[List[Generation]] = []

        for prompt in prompts:
            req["payload"]["messages"] = [{"role": "user", "content": prompt}]
            text = self._post_with_retry(requests, req)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    def _post_with_retry(self, requests, req: Dict[str, Any]) -> str:
        """POST with bounded exponential backoff on transient errors.

        Retries on network errors, timeouts, and 429/5xx (gateway hiccups),
        which are common behind Cloudflare-fronted providers like OptiLLM.
        Never includes headers in error messages (they carry the key).
        """
        import time as _time
        max_attempts = 4
        backoff = 1.5
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    req["url"],
                    headers=req["headers"],
                    json=req["payload"],
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts:
                    _time.sleep(backoff ** attempt)
                    continue
                raise RuntimeError(
                    f"Network error calling provider '{self.provider}' at "
                    f"{self.base_url} after {max_attempts} attempts: "
                    f"{exc.__class__.__name__}"
                ) from exc

            # Transient server errors -> retry with backoff
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                if attempt < max_attempts:
                    _time.sleep(backoff ** attempt)
                    continue
                raise RuntimeError(
                    f"Provider '{self.provider}' returned HTTP {resp.status_code} "
                    f"after {max_attempts} attempts: {_redact(resp.text)[:400]}"
                )

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Provider '{self.provider}' returned HTTP {resp.status_code}: "
                    f"{_redact(resp.text)[:400]}"
                )

            try:
                return self._extract_text(resp.json())
            except Exception:  # noqa: BLE001
                raise RuntimeError(
                    f"Unexpected response shape from '{self.provider}': "
                    f"{_redact(resp.text)[:400]}"
                )
        # Should not reach here, but fail safe.
        raise RuntimeError(
            f"Exhausted retries calling '{self.provider}': {last_exc}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 90.0,
    **kwargs: Any,
) -> BaseLLM:
    """Create a provider client from env/config.

    Examples:
        >>> llm = create_llm()                 # auto-detect provider from env
        >>> llm = create_llm("optillm")        # force OptiLLM (mistral-small)
        >>> llm = create_llm("optillm", model="gpt-4o-mini")
        >>> llm = create_llm("groq", temperature=0.2)
    """
    cfg = resolve_provider_config(provider, api_key, base_url, model)
    secret = SecretStr(cfg["api_key"]) if SecretStr is not None else cfg["api_key"]
    return UniversalLLM(
        provider=cfg["provider"],
        protocol=cfg["protocol"],
        api_key=secret,
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


# Backwards-compatible aliases (old code imported GroqLLM).
GroqLLM = UniversalLLM
