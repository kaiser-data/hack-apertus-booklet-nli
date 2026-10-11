"""Runtime configuration, read from the environment variables the Hack Apertus template prescribes."""

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("BOOKLET_NLI_DATA", Path(__file__).resolve().parents[2] / "data"))
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class LLMConfig:
    name: str
    base_url: str
    api_key: str


def llm_config() -> LLMConfig:
    return LLMConfig(
        name=os.environ.get("LLM_NAME", "swiss-ai/Apertus-v1.5-8B"),
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    )
