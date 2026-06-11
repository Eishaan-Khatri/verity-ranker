"""
Central config loader for ai-hiring-ranker.

Reads configs/v2/models.yaml and other YAML configs.
Provides typed config objects used by all agents.
Falls back to safe defaults if a config file is missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Optional YAML support — falls back to manual parsing if pyyaml not installed
try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

ROOT = Path(__file__).resolve().parents[3]   # repo root
CONFIGS_DIR = ROOT / "configs" / "v2"


# ---------------------------------------------------------------------------
# Typed config models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536


class HyDEConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    num_profiles: int = 3
    temperature: float = 0.7


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    hyde: HyDEConfig = Field(default_factory=HyDEConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if not _HAS_YAML:
        raise ImportError(
            "pyyaml is required to load config files. "
            "Install it with: pip install pyyaml"
        )
    with path.open(encoding="utf-8") as fh:
        return _yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_models_config() -> ModelsConfig:
    """Load and cache models.yaml. Injects API key from environment."""
    raw = _load_yaml(CONFIGS_DIR / "models.yaml")
    config = ModelsConfig(**raw)

    # Inject API key from environment — never hard-code credentials
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_RANKER_API_KEY")
    config.llm.api_key = api_key

    return config


def get_llm_config() -> LLMConfig:
    return get_models_config().llm


def get_embedding_config() -> EmbeddingConfig:
    return get_models_config().embeddings


def get_hyde_config() -> HyDEConfig:
    return get_models_config().hyde
