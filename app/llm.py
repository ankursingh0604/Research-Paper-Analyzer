"""
Thin wrapper around OpenAI that:
  - Forces structured JSON output matching a Pydantic schema (native
    support via client.beta.chat.completions.parse - no manual schema
    wrangling needed, unlike Gemini)
  - Retries transient failures (rate limits, timeouts, server errors)
    with backoff
  - Raises a clear, typed error on permanent failure so the graph can
    log it and let the Review Agent / Boss Agent decide what to do
"""
from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMCallError(Exception):
    """Raised when the LLM call fails permanently after retries, or returns unparseable output."""


class RetriableAPIError(Exception):
    pass


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMCallError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your OpenAI API key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RetriableAPIError),
)
def _call_openai(prompt: str, model_name: str, schema: Type[T]) -> T:
    client = _get_client()
    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,  # Pydantic model passed directly - OpenAI builds + validates the schema for us
            temperature=0.3,
        )
    except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as exc:
        logger.warning("Transient OpenAI API error, will retry: %s", exc)
        raise RetriableAPIError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - anything else is treated as permanent
        raise LLMCallError(f"OpenAI API call failed: {exc}") from exc

    choice = completion.choices[0]

    if choice.message.refusal:
        raise LLMCallError(f"OpenAI refused the request: {choice.message.refusal}")

    if choice.message.parsed is None:
        raise LLMCallError("OpenAI returned no parsed content (empty or malformed response)")

    return choice.message.parsed


def structured_call(prompt: str, schema: Type[T], model_name: str | None = None) -> T:
    """Call OpenAI and parse the response into `schema`. Raises LLMCallError on failure."""
    model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        return _call_openai(prompt, model_name, schema)
    except RetriableAPIError as exc:
        raise LLMCallError(f"OpenAI API failed after retries: {exc}") from exc
    except ValidationError as exc:
        raise LLMCallError(f"Could not parse OpenAI response into {schema.__name__}: {exc}") from exc