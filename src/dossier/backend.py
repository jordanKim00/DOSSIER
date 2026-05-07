from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .common import extract_json_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Local Qwen backend (Transformers)
# ---------------------------------------------------------------------------
META_COGNITIVE_SRC = Path("/workspace/meta-cognitive-RAG/src")
if str(META_COGNITIVE_SRC) not in sys.path:
    sys.path.insert(0, str(META_COGNITIVE_SRC))


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str = "/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct",
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 20,
        load_in_4bit: Optional[bool] = None,
    ) -> None:
        from meta_cognitive_rag.local_backend import LocalTransformersBackend, LocalTransformersConfig  # type: ignore
        if load_in_4bit is None:
            load_in_4bit = os.getenv(
                "DOSSIER_LOAD_IN_4BIT",
                os.getenv("LAMBO_LOAD_IN_4BIT", "1"),
            ).strip() not in {"0", "false", "False"}
        env_memory = os.getenv("DOSSIER_PER_GPU_MAX_MEMORY_GIB", "").strip() or os.getenv("LAMBO_PER_GPU_MAX_MEMORY_GIB", "").strip()
        if env_memory:
            try:
                per_gpu_max_memory_gib = int(env_memory)
            except Exception:
                pass
        self.config = LocalTransformersConfig(
            model_id="Qwen/Qwen2.5-32B-Instruct",
            model_dir=model_dir,
            max_output_tokens=max_output_tokens,
            max_input_tokens=max_input_tokens,
            compute_dtype="float16",
            load_in_4bit=load_in_4bit,
            per_gpu_max_memory_gib=per_gpu_max_memory_gib,
        )
        self.backend = LocalTransformersBackend(self.config)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.config.max_output_tokens,
            metadata=metadata or {},
        ).text

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        raw_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw_text)
        if payload is None:
            repair_prompt = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer with strict JSON only. "
                + "Do not include markdown fences or commentary."
            )
            raw_text = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw_text)
        return payload, raw_text


# ---------------------------------------------------------------------------
# Gemini API backend
# ---------------------------------------------------------------------------
class GeminiClient:
    """Google Gemini API client with the same interface as QwenLocalClient."""

    MAX_RETRIES = 10
    INITIAL_BACKOFF = 10.0
    CALL_DELAY = 7.0  # seconds between API calls for free-tier RPM limit

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import google.generativeai as genai

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in .env or pass api_key= to GeminiClient()."
            )
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._last_call_time = 0.0

    def _call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt if system_prompt else None,
        )
        gen_config = self._genai.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        # Enforce minimum delay between API calls
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self.CALL_DELAY:
            time.sleep(self.CALL_DELAY - elapsed)

        backoff = self.INITIAL_BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = model.generate_content(
                    user_prompt,
                    generation_config=gen_config,
                )
                self._last_call_time = time.time()
                return response.text
            except Exception as exc:
                err_str = str(exc).lower()
                retryable = any(
                    kw in err_str
                    for kw in ("429", "resource_exhausted", "quota", "rate", "500", "503", "unavailable", "deadline")
                )
                if retryable and attempt < self.MAX_RETRIES:
                    print(
                        f"  [Gemini] retry {attempt}/{self.MAX_RETRIES} after {backoff:.0f}s: {exc}",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self._call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens or 2048,
            temperature=temperature,
            metadata=metadata,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        raw_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw_text)
        if payload is None:
            repair_prompt = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer with strict JSON only. "
                + "Do not include markdown fences or commentary."
            )
            raw_text = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw_text)
        return payload, raw_text


# ---------------------------------------------------------------------------
# OpenAI API backend
# ---------------------------------------------------------------------------
class OpenAIClient:
    """OpenAI API client with the same interface as QwenLocalClient."""

    MAX_RETRIES = 8
    INITIAL_BACKOFF = 5.0

    @staticmethod
    def _env_int(name: str) -> Optional[int]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "").strip() or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key and self.base_url:
            self.api_key = "EMPTY"
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set it in .env or pass api_key= to OpenAIClient()."
            )
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        timeout_env = os.getenv("OPENAI_TIMEOUT_SECONDS", "").strip()
        if timeout_env:
            client_kwargs["timeout"] = float(timeout_env)
        self._client = OpenAI(**client_kwargs)
        # Qwen3 thinking-model toggle (vLLM OpenAI-compat). Set
        # OPENAI_ENABLE_THINKING=false to disable <think> output so JSON fits
        # inside max_tokens.
        temp_env = os.getenv("OPENAI_DEFAULT_TEMPERATURE", "").strip()
        self._default_temperature = float(temp_env) if temp_env else None

    def _call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self._default_temperature is not None:
            temperature = self._default_temperature

        module = (metadata or {}).get("module", "")
        module_env_name = ""
        legacy_module_env_name = ""
        if module:
            module_env_name = (
                "DOSSIER_"
                + "".join(ch if ch.isalnum() else "_" for ch in str(module)).upper()
                + "_MAX_OUTPUT_TOKENS"
            )
            legacy_module_env_name = (
                "LAMBO_"
                + "".join(ch if ch.isalnum() else "_" for ch in str(module)).upper()
                + "_MAX_OUTPUT_TOKENS"
            )
        override_tokens = (
            self._env_int(module_env_name) if module_env_name else None
        ) or (
            self._env_int(legacy_module_env_name) if legacy_module_env_name else None
        ) or self._env_int("DOSSIER_OPENAI_MAX_OUTPUT_TOKENS") or self._env_int("LAMBO_OPENAI_MAX_OUTPUT_TOKENS")
        if override_tokens is not None:
            max_output_tokens = override_tokens

        thinking_modules = set(
            filter(None, os.getenv("OPENAI_THINKING_MODULES", "").split(","))
        )
        use_thinking = module in thinking_modules if thinking_modules else False
        merge_thinking = use_thinking

        backoff = self.INITIAL_BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_output_tokens,
                    "temperature": temperature,
                }
                if not use_thinking:
                    create_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                response = self._client.chat.completions.create(**create_kwargs)
                msg = response.choices[0].message
                content = msg.content or ""
                if merge_thinking:
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
                    if reasoning:
                        content = f"<think>{reasoning}</think>\n{content}"
                return content
            except Exception as exc:
                err_str = str(exc).lower()
                retryable = any(
                    kw in err_str
                    for kw in ("429", "rate", "quota", "500", "502", "503", "overloaded", "timeout", "timed out", "readtimeout")
                )
                if retryable and attempt < self.MAX_RETRIES:
                    print(
                        f"  [OpenAI] retry {attempt}/{self.MAX_RETRIES} after {backoff:.0f}s: {exc}",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self._call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens or 2048,
            temperature=temperature,
            metadata=metadata,
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        raw_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw_text)
        if payload is None:
            repair_prompt = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer with strict JSON only. "
                + "Do not include markdown fences or commentary."
            )
            raw_text = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw_text)
        return payload, raw_text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_client_cache: dict[str, Any] = {}


def get_default_client(backend: str = "local") -> QwenLocalClient | GeminiClient | OpenAIClient:
    """Return a cached LLM client.

    Args:
        backend: "local" for QwenLocalClient, "gemini" for GeminiClient,
                 "openai" for OpenAIClient.
    """
    if backend in _client_cache:
        return _client_cache[backend]

    if backend == "gemini":
        client = GeminiClient()
    elif backend == "openai":
        client = OpenAIClient()
    else:
        client = QwenLocalClient()

    _client_cache[backend] = client
    return client
