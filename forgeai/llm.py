"""Provider-abstracted LLMClient with retries, fallback chain, disk cache, rate limiter, and JSON repair."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, ValidationError
import yaml

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when all configured LLM providers fail."""


class BudgetExceededError(RuntimeError):
    """Raised when token or wall-clock budget is exhausted."""


@dataclass
class ProviderTarget:
    """A provider + model endpoint specification."""

    provider: str
    model: str
    family: str = ""


@dataclass
class TokenBucket:
    """Simple requests-per-minute token bucket rate limiter."""

    rpm: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = max(1.0, float(self.rpm))
        self.last_refill = time.monotonic()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        rate_per_sec = self.rpm / 60.0
        while True:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(float(self.rpm), self.tokens + elapsed * rate_per_sec)
            self.last_refill = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            wait_s = (1.0 - self.tokens) / max(rate_per_sec, 1e-6)
            time.sleep(min(wait_s, 5.0))


@dataclass
class CallMetadata:
    """Telemetry returned alongside LLM responses."""

    provider: str
    model: str
    family: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    cached: bool


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml from explicit path, CWD, or package parent."""
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(Path(__file__).resolve().parent.parent / "config.yaml")
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def _extract_json_block(text: str) -> str:
    """Extract a JSON object or array from raw model output."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()
    # Find outermost braces or brackets if extra prose surrounds JSON.
    first_obj = stripped.find("{")
    first_arr = stripped.find("[")
    starts = [idx for idx in (first_obj, first_arr) if idx != -1]
    if not starts:
        return stripped
    start = min(starts)
    open_ch = stripped[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    return stripped[start:]


class LLMClient:
    """Unified LLM client with retries, fallbacks, caching, rate limits, and JSON repair."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        use_cache: bool | None = None,
        cache_dir: str | Path | None = None,
        mock_handler: Any | None = None,
    ) -> None:
        self.config = config if config is not None else load_config()
        llm_cfg = self.config.get("llm", {})
        env_no_cache = os.getenv("FORGEAI_NO_CACHE", "").lower() in {"1", "true", "yes"}
        if use_cache is not None:
            self.use_cache = use_cache
        elif env_no_cache:
            self.use_cache = False
        else:
            self.use_cache = bool(llm_cfg.get("cache_enabled", True))

        self.cache_dir = Path(cache_dir or llm_cfg.get("cache_dir", ".forgeai_cache/llm"))
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = float(llm_cfg.get("timeout_seconds", 45))
        self.max_retries = int(llm_cfg.get("max_retries", 3))
        self.json_repair_retries = int(llm_cfg.get("json_repair_retries", 2))
        self.mock_handler = mock_handler

        rpm_cfg = llm_cfg.get("rate_limits_rpm", {})
        self._buckets: dict[str, TokenBucket] = {
            "groq": TokenBucket(float(rpm_cfg.get("groq", 25))),
            "gemini": TokenBucket(float(rpm_cfg.get("gemini", 14))),
            "openrouter": TokenBucket(float(rpm_cfg.get("openrouter", 15))),
            "ollama": TokenBucket(float(rpm_cfg.get("ollama", 120))),
        }

        budgets = self.config.get("budgets", {})
        self.max_tokens_budget = int(budgets.get("max_tokens_per_run", 450000))
        self.max_minutes_budget = float(budgets.get("max_minutes_per_run", 20))
        self.start_time = time.monotonic()

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.call_count = 0
        self.cache_hits = 0
        self.last_call: Optional[CallMetadata] = None

    def reset_counters(self) -> None:
        self.start_time = time.monotonic()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_latency_s = 0.0
        self.call_count = 0
        self.cache_hits = 0

    def check_budget(self) -> None:
        if self.total_tokens > self.max_tokens_budget:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.total_tokens} > {self.max_tokens_budget}"
            )
        elapsed_min = (time.monotonic() - self.start_time) / 60.0
        if elapsed_min > self.max_minutes_budget:
            raise BudgetExceededError(
                f"Time budget exceeded: {elapsed_min:.1f}m > {self.max_minutes_budget:.1f}m"
            )

    def get_role_targets(self, model_role: str) -> list[ProviderTarget]:
        roles = self.config.get("roles", {})
        role_cfg = roles.get(model_role) or roles.get("factory") or {}
        targets: list[ProviderTarget] = []
        primary = role_cfg.get("primary")
        if primary:
            targets.append(ProviderTarget(**primary))
        for fb in role_cfg.get("fallbacks", []):
            targets.append(ProviderTarget(**fb))

        # Enforce distinct family between judge and agent primary whenever possible.
        if model_role == "judge":
            agent_primary = roles.get("agent", {}).get("primary", {})
            agent_family = agent_primary.get("family", "")
            agent_model = agent_primary.get("model", "")
            filtered = [
                t
                for t in targets
                if (t.family and t.family != agent_family) or (t.model != agent_model)
            ]
            if filtered:
                targets = filtered
        return targets

    def check_providers(self) -> dict[str, bool]:
        """Check which free providers have keys or local daemon reachable."""
        status = {
            "groq": bool(os.getenv("GROQ_API_KEY", "").strip()),
            "gemini": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "ollama": False,
        }
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=1.5)
            status["ollama"] = resp.status_code == 200
        except Exception:
            status["ollama"] = False
        return status

    def _provider_has_credentials(self, provider: str) -> bool:
        if self.mock_handler is not None:
            return True
        if provider == "groq":
            return bool(os.getenv("GROQ_API_KEY", "").strip())
        if provider == "gemini":
            return bool(os.getenv("GEMINI_API_KEY", "").strip())
        if provider == "openrouter":
            return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        if provider == "ollama":
            return True
        return False

    def _cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        schema_name: str,
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "schema": schema_name,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> Optional[dict[str, Any]]:
        if not self.use_cache:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _write_cache(self, key: str, data: dict[str, Any]) -> None:
        if not self.use_cache:
            return
        path = self.cache_dir / f"{key}.json"
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        except Exception:
            pass

    def _invoke_provider(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        expect_json: bool,
    ) -> tuple[str, int, int]:
        if self.mock_handler is not None:
            text = self.mock_handler(target, messages, temperature, expect_json)
            pt = sum(len(m.get("content", "")) for m in messages) // 4
            ct = len(text) // 4
            return text, pt, ct

        bucket = self._buckets.get(target.provider)
        if bucket is not None:
            bucket.acquire()

        if target.provider in {"groq", "openrouter"}:
            return self._call_openai_compatible(
                target, messages, temperature, max_tokens, expect_json
            )
        if target.provider == "gemini":
            return self._call_gemini(
                target, messages, temperature, max_tokens, expect_json
            )
        if target.provider == "ollama":
            return self._call_ollama(
                target, messages, temperature, max_tokens, expect_json
            )
        raise LLMError(f"Unsupported provider: {target.provider}")

    def _call_openai_compatible(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        expect_json: bool,
    ) -> tuple[str, int, int]:
        if target.provider == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if expect_json and target.provider == "groq":
            body["response_format"] = {"type": "json_object"}

        resp = httpx.post(url, headers=headers, json=body, timeout=self.timeout_seconds)
        if resp.status_code in {429, 500, 502, 503, 504}:
            raise httpx.HTTPStatusError(
                f"Transient HTTP {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        return text, pt, ct

    def _call_gemini(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        expect_json: bool,
    ) -> tuple[str, int, int]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{target.model}:generateContent?key={api_key}"
        )
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        gen_cfg: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if expect_json:
            gen_cfg["responseMimeType"] = "application/json"

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        resp = httpx.post(url, json=body, timeout=self.timeout_seconds)
        if resp.status_code in {429, 500, 502, 503, 504}:
            raise httpx.HTTPStatusError(
                f"Transient HTTP {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        pt = int(usage.get("promptTokenCount", 0))
        ct = int(usage.get("candidatesTokenCount", 0))
        return text, pt, ct

    def _call_ollama(
        self,
        target: ProviderTarget,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        expect_json: bool,
    ) -> tuple[str, int, int]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        body: dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if expect_json:
            body["format"] = "json"
        resp = httpx.post(
            f"{base_url}/api/chat", json=body, timeout=self.timeout_seconds
        )
        if resp.status_code in {429, 500, 502, 503, 504}:
            raise httpx.HTTPStatusError(
                f"Transient HTTP {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        pt = int(data.get("prompt_eval_count", 0))
        ct = int(data.get("eval_count", 0))
        return text, pt, ct

    def _call_with_fallback(
        self,
        messages: list[dict[str, str]],
        model_role: str,
        temperature: float,
        max_tokens: int,
        expect_json: bool,
        schema_name: str,
    ) -> str:
        self.check_budget()
        targets = self.get_role_targets(model_role)
        if not targets:
            raise LLMError(f"No model targets configured for role '{model_role}'")

        errors: list[str] = []
        for target in targets:
            if not self._provider_has_credentials(target.provider):
                errors.append(f"{target.provider}:{target.model} skipped (no key)")
                continue

            cache_key = self._cache_key(target.model, messages, temperature, schema_name)
            cached = self._read_cache(cache_key)
            if cached is not None:
                self.cache_hits += 1
                self.call_count += 1
                self.last_call = CallMetadata(
                    provider=target.provider,
                    model=target.model,
                    family=target.family,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_s=0.0,
                    cached=True,
                )
                return cached["text"]

            for attempt in range(self.max_retries):
                t0 = time.monotonic()
                try:
                    text, pt, ct = self._invoke_provider(
                        target, messages, temperature, max_tokens, expect_json
                    )
                    latency = time.monotonic() - t0
                    total = pt + ct
                    self.total_prompt_tokens += pt
                    self.total_completion_tokens += ct
                    self.total_tokens += total
                    self.total_latency_s += latency
                    self.call_count += 1
                    self.last_call = CallMetadata(
                        provider=target.provider,
                        model=target.model,
                        family=target.family,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=total,
                        latency_s=latency,
                        cached=False,
                    )
                    self._write_cache(cache_key, {"text": text})
                    return text
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    errors.append(
                        f"{target.provider}:{target.model} attempt {attempt + 1} HTTP {status}"
                    )
                    if status in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                        time.sleep(min(2.0**attempt, 8.0))
                        continue
                    break
                except Exception as exc:
                    errors.append(f"{target.provider}:{target.model} error: {exc}")
                    if attempt < self.max_retries - 1:
                        time.sleep(min(1.5**attempt, 4.0))
                        continue
                    break

        raise LLMError(
            f"All providers failed for role '{model_role}': " + " | ".join(errors)
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model_role: str = "factory",
        json_schema: Optional[Type[T]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> str | T:
        """Send chat messages and optionally validate/repair against a Pydantic model."""
        expect_json = json_schema is not None
        schema_name = json_schema.__name__ if json_schema is not None else ""

        working_messages = [dict(m) for m in messages]
        if json_schema is not None:
            schema_hint = json.dumps(json_schema.model_json_schema(), indent=2)
            instruction = (
                "\n\nRespond ONLY with valid JSON matching this JSON Schema "
                f"(no markdown prose outside JSON):\n{schema_hint}"
            )
            if working_messages and working_messages[-1].get("role") == "user":
                working_messages[-1]["content"] = (
                    working_messages[-1]["content"] + instruction
                )
            else:
                working_messages.append({"role": "user", "content": instruction})

        raw = self._call_with_fallback(
            working_messages,
            model_role=model_role,
            temperature=temperature,
            max_tokens=max_tokens,
            expect_json=expect_json,
            schema_name=schema_name,
        )
        if json_schema is None:
            return raw

        last_err: Optional[Exception] = None
        current_raw = raw
        for repair_idx in range(self.json_repair_retries + 1):
            extracted = _extract_json_block(current_raw)
            try:
                parsed = json.loads(extracted)
                return json_schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_err = exc
                if repair_idx >= self.json_repair_retries:
                    break
                repair_messages = working_messages + [
                    {"role": "assistant", "content": current_raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed JSON/schema validation with error:\n"
                            f"{exc}\n\n"
                            "Return ONLY corrected valid JSON matching the schema."
                        ),
                    },
                ]
                current_raw = self._call_with_fallback(
                    repair_messages,
                    model_role=model_role,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    expect_json=True,
                    schema_name=f"{schema_name}_repair_{repair_idx}",
                )

        raise LLMError(
            f"JSON validation failed after {self.json_repair_retries} repair retries "
            f"for {schema_name}: {last_err}"
        )
