"""Thin OpenAI-compatible client: works against vLLM (Modal), Ollama or any compatible server."""

import time
from dataclasses import dataclass

from openai import BadRequestError, OpenAI

from .config import LLMConfig


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM:
    def __init__(self, cfg: LLMConfig, timeout: float = 900):
        self.model = cfg.name
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout)
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
        except BadRequestError:
            if "response_format" not in kwargs:
                raise
            # Backend without structured output: fall back to prompt-only JSON.
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
