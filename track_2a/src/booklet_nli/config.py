"""Runtime configuration, read from the environment.

The official evaluation injects `BASE_URL` and `API_KEY`; the Hack Apertus template's
`LLM_BASE_URL` / `LLM_API_KEY` are still read when those are unset.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("BOOKLET_NLI_DATA", Path(__file__).resolve().parents[2] / "data"))
CACHE_DIR = DATA_DIR / "cache"
# Embedder weights; the Docker image bakes them into a read-only path at build time.
EMBEDDER_DIR = Path(os.environ.get("BOOKLET_NLI_EMBEDDERS", CACHE_DIR / "embedders"))


@dataclass(frozen=True)
class LLMConfig:
    name: str
    base_url: str
    api_key: str


def _env(*names: str, default: str) -> str:
    return next((os.environ[n] for n in names if os.environ.get(n)), default)


def llm_config() -> LLMConfig:
    return LLMConfig(
        name=_env("LLM_NAME", default="swiss-ai/Apertus-v1.5-8B"),
        base_url=_env("BASE_URL", "LLM_BASE_URL", default="http://localhost:11434/v1"),
        api_key=_env("API_KEY", "LLM_API_KEY", default="not-needed"),
    )
