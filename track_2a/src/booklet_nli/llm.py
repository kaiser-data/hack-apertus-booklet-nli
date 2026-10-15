"""Thin OpenAI-compatible client: works against vLLM (Modal), Ollama or any compatible server."""

import re
import time
from dataclasses import dataclass

from openai import BadRequestError, NotFoundError, OpenAI, PermissionDeniedError

from .config import LLMConfig

_UNSUPPORTED = re.compile(r"response_format|json_schema|guided|structured", re.IGNORECASE)
# A proxy may reject a model it does not serve with 400 or 403 instead of 404.
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
                             max_retries=4)
        self._schema_supported = True
        self._fallbacks = list(fallbacks)  # tried in order if the server does not know a model

    def served_models(self) -> list[str]:
        """Model ids the endpoint lists (empty if it does not support the listing)."""
        try:
            quick = self.client.with_options(timeout=10, max_retries=0)  # never stall the run
            return [m.id for m in quick.models.list().data]
        except Exception:  # noqa: BLE001 - listing is optional for OpenAI-compatible proxies
            return []

    def pick(self, preferred: list[str]) -> str:
        """Use the first preferred model the endpoint serves (matched case-insensitively,
        ignoring any organisation prefix); keep the others as fallbacks."""
        served = self.served_models()
        def find(want: str) -> str | None:
            tail = want.split("/")[-1].lower()
            exact = [m for m in served if m.lower() == want.lower()]
            loose = [m for m in served if m.lower().split("/")[-1] == tail]
            return (exact or loose or [None])[0]
        found = [m for m in (find(w) for w in preferred) if m]
        chosen = found or preferred
        self.model, self._fallbacks = chosen[0], list(chosen[1:])
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
            try:
                return self.client.chat.completions.create(**{**kwargs, "model": self.model})
            except (NotFoundError, PermissionDeniedError, BadRequestError) as exc:
                rejected = not isinstance(exc, BadRequestError) or (
                    _UNKNOWN_MODEL.search(str(exc)) and not _UNSUPPORTED.search(str(exc)))
                if not (rejected and self._fallbacks):
                    raise
                self.model = self._fallbacks.pop(0)  # the server does not serve this model
