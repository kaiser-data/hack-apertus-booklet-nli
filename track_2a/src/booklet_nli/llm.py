"""Thin OpenAI-compatible client: works against vLLM (Modal), Ollama or any compatible server."""

import re
import time
from dataclasses import dataclass

from openai import BadRequestError, OpenAI

from .config import LLMConfig

_UNSUPPORTED = re.compile(r"response_format|json_schema|guided|structured", re.IGNORECASE)


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM:
    def __init__(self, cfg: LLMConfig, timeout: float = 900):
        self.model = cfg.name
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout,
                             max_retries=4)
        self._schema_supported = True

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
            resp = self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            # Only a backend that rejects structured output falls back to prompt-only JSON;
            # other 400s (e.g. context length) must not change how later requests are sent.
            if "response_format" not in kwargs or not _UNSUPPORTED.search(str(exc)):
                raise
            self._schema_supported = False
            kwargs.pop("response_format")
            resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=time.perf_counter() - start,
        )
