"""Shared OpenAI-compatible clients for the Kloudeks endpoint.

Exposes:
  - sync helpers `chat`, `embed`, `embed_batch` for ad-hoc / interactive use
  - async helpers `achat`, `aembed`, `aembed_batch` and `gather_*` utilities
    so modules can issue many concurrent API calls with asyncio.gather and a
    bounded semaphore.
"""
import asyncio
import os
from functools import lru_cache
from typing import Awaitable, Callable, Iterable, Sequence, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

_BASE_URL = "https://mia.csp.kloudeks.com/v1"
_LLM_MODEL = "gpt-oss-120b"
_EMBED_MODEL = "qwen3-embedding-8b"
_EMBED_DIM = 4096
DEFAULT_CONCURRENCY = 16

T = TypeVar("T")


def _api_key() -> str:
    api_key = os.getenv("KKB_API")
    if not api_key:
        raise RuntimeError("KKB_API not set in environment / .env")
    return api_key


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=_BASE_URL, api_key=_api_key())


@lru_cache(maxsize=1)
def _aclient() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=_BASE_URL, api_key=_api_key())


# ---------- sync ----------

def chat(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _client().chat.completions.create(
        model=_LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def embed(text: str) -> list[float]:
    resp = _client().embeddings.create(model=_EMBED_MODEL, input=text, encoding_format="float")
    return resp.data[0].embedding


def embed_batch(texts: Iterable[str]) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []
    resp = _client().embeddings.create(model=_EMBED_MODEL, input=texts, encoding_format="float")
    return [d.embedding for d in resp.data]


# ---------- async ----------

async def achat(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await _aclient().chat.completions.create(
        model=_LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


async def aembed(text: str) -> list[float]:
    resp = await _aclient().embeddings.create(model=_EMBED_MODEL, input=text, encoding_format="float")
    return resp.data[0].embedding


async def aembed_batch(texts: Iterable[str]) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []
    resp = await _aclient().embeddings.create(model=_EMBED_MODEL, input=texts, encoding_format="float")
    return [d.embedding for d in resp.data]


async def gather_bounded(
    awaitables: Sequence[Awaitable[T]],
    concurrency: int = DEFAULT_CONCURRENCY,
    on_done: Callable[[int], None] | None = None,
) -> list[T]:
    """asyncio.gather with a semaphore so we don't open thousands of sockets.

    Optional `on_done(i)` callback fires per completed task (in completion order)
    for progress reporting.
    """
    sem = asyncio.Semaphore(concurrency)
    done_counter = {"n": 0}

    async def _run(i: int, aw: Awaitable[T]) -> T:
        async with sem:
            result = await aw
        done_counter["n"] += 1
        if on_done is not None:
            on_done(done_counter["n"])
        return result

    return await asyncio.gather(*[_run(i, aw) for i, aw in enumerate(awaitables)])


EMBED_DIM = _EMBED_DIM
