"""
LLM Adapters: Plug-and-play connectors for any LLM provider.

Each adapter returns a (llm_fn, embed_fn) tuple ready for Cortex.

Embedding priority:
    1. sentence-transformers (all-MiniLM-L6-v2, 22MB) — default when installed
    2. Provider-native API embeddings (OpenAI, Ollama)
    3. _simple_embed() — character n-gram fallback, dev/edge only
"""
import openai
import json
import math
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Callable
import requests
import anthropic
from sentence_transformers import SentenceTransformer as _SentenceTransformer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Semantic Embedder — sentence-transformers (preferred default)
# pip install sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────

try:
   
    _ST_MODEL_NAME = "all-MiniLM-L6-v2"
    _st_model = None  # lazy-loaded on first call

    def _get_st_model():
        global _st_model
        if _st_model is None:
            logger.info(f"[Llm] Loading sentence-transformers model '{_ST_MODEL_NAME}' (first call)...")
            _st_model = _SentenceTransformer(_ST_MODEL_NAME)
        return _st_model

    def _semantic_embed(text: str) -> List[float]:
        """384-dim semantic embedding via sentence-transformers (all-MiniLM-L6-v2)."""
        return _get_st_model().encode(text, normalize_embeddings=True).tolist()

    _ST_AVAILABLE = True
    logger.debug("[Llm] sentence-transformers available — semantic embeddings enabled.")

except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "[Llm] sentence-transformers not installed. "
        "Falling back to n-gram embedder (not suitable for production). "
        "Install with: pip install sentence-transformers"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Embedder — character n-grams (dev / offline / edge only)
# ─────────────────────────────────────────────────────────────────────────────

def _simple_embed(text: str, dim: int = 128) -> List[float]:
    """
    Lightweight deterministic text embedder using character n-grams.
    No dependencies. NOT for production — semantic similarity is character-level
    only. Two sentences with identical meaning but different words score ~0.

    Use cases: offline edge deployments, unit tests, CI environments.
    """
    text = text.lower().strip()
    vec = [0.0] * dim

    for n in [2, 3]:
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0

    for word in text.split():
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 2.0

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]

    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Default embed_fn — semantic if available, n-gram otherwise
# ─────────────────────────────────────────────────────────────────────────────

_default_embed: Callable = _semantic_embed if _ST_AVAILABLE else _simple_embed


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Adapter
# ─────────────────────────────────────────────────────────────────────────────

def openai_adapter(
    api_key: str,
    model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
) -> Tuple[Callable, Callable]:
    """
    Returns (llm_fn, embed_fn) for OpenAI.

    Usage:
        llm_fn, embed_fn = openai_adapter(api_key="sk-...")
        brain = Cortex(llm_fn=llm_fn, embed_fn=embed_fn)
    """
    client = openai.OpenAI(api_key=api_key)

    def llm_fn(messages: List[Dict], system: str) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
        )
        return response.choices[0].message.content

    def embed_fn(text: str) -> List[float]:
        response = client.embeddings.create(
            input=text,
            model=embed_model,
        )
        return response.data[0].embedding

    return llm_fn, embed_fn


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic Adapter
# ─────────────────────────────────────────────────────────────────────────────

def anthropic_adapter(
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> Tuple[Callable, Callable]:
    """
    Returns (llm_fn, embed_fn) for Anthropic Claude.

    Anthropic has no native embedding API, so embed_fn uses sentence-transformers
    (all-MiniLM-L6-v2) when installed, falling back to the n-gram embedder.
    For highest-quality embeddings, use openai_adapter() or pass a custom embed_fn
    to generic_adapter().

    Usage:
        llm_fn, embed_fn = anthropic_adapter(api_key="sk-ant-...")
        brain = Cortex(llm_fn=llm_fn, embed_fn=embed_fn)
    """
    client = anthropic.Anthropic(api_key=api_key)

    def llm_fn(messages: List[Dict], system: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    return llm_fn, _default_embed


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Adapter (local models)
# ─────────────────────────────────────────────────────────────────────────────

def ollama_adapter(
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> Tuple[Callable, Callable]:
    """
    Returns (llm_fn, embed_fn) for Ollama (local models).

    embed_fn tries Ollama's native /api/embeddings endpoint first, then falls
    back to sentence-transformers, then n-gram.

    Usage:
        llm_fn, embed_fn = ollama_adapter(model="llama3")
        brain = Cortex(llm_fn=llm_fn, embed_fn=embed_fn)
    """

    def llm_fn(messages: List[Dict], system: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def embed_fn(text: str) -> List[float]:
        try:
            resp = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception:
            # Fall back to sentence-transformers or n-gram
            return _default_embed(text)

    return llm_fn, embed_fn


# ─────────────────────────────────────────────────────────────────────────────
# Custom / Generic Adapter
# ─────────────────────────────────────────────────────────────────────────────

def generic_adapter(
    llm_fn: Callable,
    embed_fn: Optional[Callable] = None,
) -> Tuple[Callable, Callable]:
    """
    Wrap any custom LLM function.

    Args:
        llm_fn:   Callable(messages: list[dict], system: str) -> str
        embed_fn: Optional Callable(text: str) -> list[float]
                  Defaults to sentence-transformers if installed, n-gram otherwise.
    """
    return llm_fn, embed_fn or _default_embed