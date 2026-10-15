"""Thin OpenAI-compatible client: works against vLLM (Modal), Ollama or any compatible server."""

import re
import threading
import time
from dataclasses import dataclass

from openai import (
    APIStatusError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    UnprocessableEntityError,
)

from .config import LLMConfig

_UNSUPPORTED = re.compile(r"response_format|json_schema|guided|structured", re.IGNORECASE)
# A proxy may reject a model it does not serve with 400 or 403 instead of 404.
_CONTEXT = re.compile(r"context length|maximum context|too many tokens|too long", re.IGNORECASE)
_UNKNOWN_MODEL = re.compile(
    r"model\b.{0,80}\b(does not exist|not found|unknown|invalid|not available|not supported|"
    r"no such|not allowed|not permitted)|\b(unknown|invalid|unsupported) model", re.IGNORECASE)


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM:
    def __init__(self, cfg: LLMConfig, timeout: float = 900, fallbacks: tuple[str, ...] = ()):
        self.model = cfg.name
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout,
                             max_retries=3)
        self._schema_supported = True
        self._fallbacks = list(fallbacks)  # tried in order if the server does not know a model
        self._switch = threading.Lock()

    def served_models(self) -> list[str]:
        """Model ids the endpoint lists (empty if it does not support the listing)."""
        try:
            quick = self.client.with_options(timeout=10, max_retries=0)  # never stall the run
            return [m.id for m in quick.models.list().data]
        except Exception:  # noqa: BLE001 - listing is optional for OpenAI-compatible proxies
            return []

    def pick(self, preferred: list[str]) -> str:
        """Use the first preferred model the endpoint serves; keep the rest as fallbacks.

        Matching ignores case and any organisation prefix and accepts suffixes such as
        "-Instruct" ("Apertus-v1.5-70B" matches "swiss-ai/apertus-v1.5-70b-instruct"). Preferred
        models the listing does not show stay at the end of the list, and any other listed
        Apertus model after them, so an incomplete listing never removes a candidate.
        """
        served = self.served_models()

        def find(want: str) -> str | None:
            tail = want.split("/")[-1].lower()
            ranked = ([m for m in served if m.lower() == want.lower()]
                      + [m for m in served if m.lower().split("/")[-1] == tail]
                      + [m for m in served if tail in m.lower()])
            return ranked[0] if ranked else None

        found = [m for m in dict.fromkeys(find(w) for w in preferred) if m]
        rest = [w for w in preferred if w not in found and find(w) is None]
        others = [m for m in served if "apertus" in m.lower() and m not in found]
        chosen = list(dict.fromkeys(found + rest + others))
        self.model, self._fallbacks = chosen[0], chosen[1:]
        return self.model

    def complete(self, messages: list[dict], max_tokens: int, json_schema: dict | None = None) -> Completion:
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                        "temperature": 0.0}
        if json_schema and self._schema_supported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": json_schema, "strict": True},
            }
        start = time.perf_counter()
        try:
            resp = self._create(kwargs)
        except BadRequestError as exc:
            # Only a backend that rejects structured output falls back to prompt-only JSON;
            # other 400s (e.g. context length) must not change how later requests are sent.
            if "response_format" not in kwargs or not _UNSUPPORTED.search(str(exc)):
                raise
            self._schema_supported = False
            kwargs.pop("response_format")
            resp = self._create(kwargs)
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=time.perf_counter() - start,
        )

    def _create(self, kwargs: dict):
        while True:
            model = self.model
            try:
                return self.client.chat.completions.create(**{**kwargs, "model": model})
            except (NotFoundError, PermissionDeniedError, BadRequestError,
                    UnprocessableEntityError) as exc:
                if not (_rejects_model(exc) and self._fallbacks):
                    raise
                with self._switch:  # one switch per failed model, whatever the thread
                    if self.model == model and self._fallbacks:
                        self.model = self._fallbacks.pop(0)


def _rejects_model(exc: APIStatusError) -> bool:
    """True if the server says it does not serve the requested model. Only the error message is
    read (not the error body: its "type" field, e.g. "invalid_request_error", says nothing about
    the model), and a context-length error never counts."""
    message = getattr(exc, "message", "") or ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        message = f"{message} {inner.get('message', '') if isinstance(inner, dict) else ''}"
    if _CONTEXT.search(message) or _UNSUPPORTED.search(message):
        return False
    return isinstance(exc, NotFoundError) or bool(_UNKNOWN_MODEL.search(message))
