"""
LLM provider abstraction layer.

All agents call `chat_completion()` and `structured_completion()` from here.
Swapping from OpenAI → Anthropic → local model only requires changing
configs/v2/models.yaml — no agent code changes needed.

Structured output strategy:
  - OpenAI models with json_schema support use the `response_format` parameter
    for guaranteed valid JSON (no retry needed).
  - Fallback: prompt-level JSON instruction + manual parse + Pydantic validation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from .config import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Low-level chat call
# ---------------------------------------------------------------------------


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    config: Optional[LLMConfig] = None,
    response_format: Optional[dict] = None,
) -> str:
    """Call the configured LLM and return the raw string response.

    Args:
        system_prompt: System-level instructions for the model.
        user_prompt:   User-level content (the JD text, resume text, etc.)
        config:        Override the global LLM config for this call.
        response_format: Optional OpenAI response_format dict (e.g. json_object).

    Returns:
        The model's text response as a plain string.

    Raises:
        RuntimeError: If the API key is missing or the API call fails.
    """
    cfg = config or get_llm_config()

    if not cfg.api_key:
        raise RuntimeError(
            "No API key found. Set the OPENAI_API_KEY environment variable.\n"
            "Example: $env:OPENAI_API_KEY = 'sk-...'"
        )

    if cfg.provider == "openai":
        return _openai_chat(system_prompt, user_prompt, cfg, response_format)
    else:
        raise NotImplementedError(
            f"Provider '{cfg.provider}' is not yet implemented. "
            "Supported: openai. Add your provider in llm_provider.py."
        )


def _openai_chat(
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    response_format: Optional[dict],
) -> str:
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=cfg.api_key)

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if response_format:
        kwargs["response_format"] = response_format

    logger.debug("LLM call → model=%s", cfg.model)
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    logger.debug("LLM response length: %d chars", len(content))
    return content


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> str:
    """Pull the first JSON object or array out of a freeform string."""
    # Try a markdown ```json ... ``` fence first
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Try a bare JSON object
    bare = re.search(r"(\{.*\})", text, re.DOTALL)
    if bare:
        return bare.group(1)
    return text  # let json.loads raise the error with context


def structured_completion(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    *,
    config: Optional[LLMConfig] = None,
    max_retries: int = 2,
) -> T:
    """Call the LLM and parse + validate the response as a Pydantic model.

    Uses OpenAI's `response_format: {type: json_object}` when available
    for reliable JSON output, then validates against *schema*.

    Args:
        system_prompt: Instruction prompt (should include JSON schema hint).
        user_prompt:   The actual content to analyse.
        schema:        Pydantic model class to parse the response into.
        config:        Override LLM config.
        max_retries:   How many times to retry on parse/validation failure.

    Returns:
        A validated instance of *schema*.

    Raises:
        ValueError: If all retries are exhausted without a valid response.
    """
    cfg = config or get_llm_config()
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = chat_completion(
                system_prompt,
                user_prompt,
                config=cfg,
                response_format={"type": "json_object"},
            )
            data = json.loads(_extract_json_block(raw))
            return schema.model_validate(data)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "structured_completion attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                # On retry, be more explicit in the prompt
                user_prompt = (
                    user_prompt
                    + f"\n\n[IMPORTANT: Previous attempt failed with: {exc}. "
                    "Return ONLY valid JSON matching the schema exactly.]"
                )

    raise ValueError(
        f"LLM structured output failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
