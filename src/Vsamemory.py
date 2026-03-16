"""
VSAMemory: Hyperdimensional Computing (HDC) Memory Kernel (Phase 2 of Aegis-1).

Hyperdimensional computing (HDC) — also called Vector Symbolic Architectures
(VSA) — represents concepts as extremely high-dimensional binary vectors.
The key properties that make this useful for associative memory:

    Binding   : XOR(a, b) creates a composite vector dissimilar to both a and b
    Bundling  : majority_vote([a, b, c]) creates a vector similar to all inputs
    Similarity: cosine distance in high-D space approximates conceptual similarity

This module implements a full HDC associative memory store with:
    - Automatic encoding of text → hypervectors via n-gram tokenisation
    - Associative retrieval: query by text, get similar stored memories
    - Temporal decay: older memories gradually fade
    - Capacity management: weakest traces pruned when at capacity
    - Bundle cache: topic-level superposition vectors for cluster queries
    - Logical operations: AND / OR / NOT on memory vectors
"""

import time
import uuid
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HDMemoryTrace:
    """A single memory entry in the HDC store."""
    id: str
    label: str
    hypervector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    activations: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def fire(self) -> None:
        """Strengthen trace on retrieval (HDC potentiation)."""
        self.activations   += 1
        self.last_accessed  = time.time()
        self.weight         = min(self.weight * 1.05, 10.0)

    def decay(self, hours_elapsed: float, decay_rate: float = 0.02) -> None:
        """Weaken trace with time (HDC depression)."""
        self.weight = max(self.weight * math.exp(-decay_rate * hours_elapsed), 0.01)


@dataclass
class QueryResult:
    """Result of a VSA memory query."""
    label: str
    text: str
    similarity: float
    weight: float
    metadata: Dict[str, Any]
    trace_id: str

    def __repr__(self) -> str:
        return (
            f"QueryResult(label={self.label!r}, "
            f"sim={self.similarity:.3f}, weight={self.weight:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HDC Primitives
# ─────────────────────────────────────────────────────────────────────────────

def _random_hv(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random binary hypervector {0, 1}^D with ~50% density."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def _bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding via XOR. Recoverable: bind(bind(a, b), b) ≈ a"""
    return np.bitwise_xor(a, b)


def _bundle(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Bundle (superposition) via weighted majority vote."""
    if not vectors:
        raise ValueError("Cannot bundle empty list.")
    if len(vectors) == 1:
        return vectors[0].copy()
    weights = weights or [1.0] * len(vectors)
    stack   = np.stack(vectors, axis=0).astype(np.float32)
    w       = np.array(weights, dtype=np.float32)[:, np.newaxis]
    return ((stack * w).sum(axis=0) >= w.sum() / 2.0).astype(np.uint8)


def _cosine_hd(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two binary hypervectors."""
    a_f  = a.astype(np.float32) * 2 - 1
    b_f  = b.astype(np.float32) * 2 - 1
    dot  = float(np.dot(a_f, b_f))
    norm = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    return dot / norm if norm > 0 else 0.0


def _permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Cyclic permutation — encodes sequential position."""
    return np.roll(v, shift)


# ─────────────────────────────────────────────────────────────────────────────
# Codebook
# ─────────────────────────────────────────────────────────────────────────────

class HDCodebook:
    """Maps string tokens to stable random hypervectors."""

    def __init__(self, dim: int):
        self.dim   = dim
        self._book: Dict[str, np.ndarray] = {}
        self._rng  = np.random.default_rng(seed=2024)

    def encode(self, token: str) -> np.ndarray:
        if token not in self._book:
            self._book[token] = self._rng.integers(0, 2, size=self.dim, dtype=np.uint8)
        return self._book[token]

    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        return _bundle([_permute(self.encode(t), shift=i) for i, t in enumerate(tokens)])

    def text_to_hv(self, text: str) -> np.ndarray:
        tokens = self._tokenise(text)
        if not tokens:
            return _random_hv(self.dim)
        unigrams = [self.encode(t) for t in tokens]
        bigrams  = [
            _bind(self.encode(tokens[i]), _permute(self.encode(tokens[i + 1])))
            for i in range(len(tokens) - 1)
        ]
        return _bundle(unigrams + bigrams)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        import re
        return re.findall(r"\b\w+\b", text.lower())

    @property
    def vocab_size(self) -> int:
        return len(self._book)


# ─────────────────────────────────────────────────────────────────────────────
# VSA Memory Store
# ─────────────────────────────────────────────────────────────────────────────

class VSAMemory:
    """
    Hyperdimensional associative memory for Aegis-1.

    Supports:
    - Binding   : associate (key, value) concept pairs
    - Bundling  : create superposition memories (topic clusters)
    - Querying  : associative retrieval by similarity
    - Decay     : temporal salience decay (older memories fade)
    - Pruning   : remove weak traces to stay within capacity
    - Logical   : AND / OR / NOT operations on memory vectors

    Usage:
        mem = VSAMemory(dim=10_000)
        mem.store("sky_is_blue", "The sky appears blue due to Rayleigh scattering.")
        results = mem.query("What colour is the sky?", top_k=3)
    """

    def __init__(
        self,
        dim: int = 10_000,
        capacity: int = 5_000,
        similarity_threshold: float = 0.55,
        decay_enabled: bool = True,
        prune_threshold: float = 0.05,
        verbose: bool = False,
    ):
        self.dim                  = dim
        self.capacity             = capacity
        self.similarity_threshold = similarity_threshold
        self.decay_enabled        = decay_enabled
        self.prune_threshold      = prune_threshold
        self.verbose              = verbose

        self._codebook               = HDCodebook(dim=dim)
        self._store: Dict[str, HDMemoryTrace]   = {}
        self._text_cache: Dict[str, str]         = {}
        self._bundle_cache: Dict[str, np.ndarray] = {}

        self._store_count = 0
        self._query_count = 0

    # ─────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────

    def store(
        self,
        label: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
        force_new: bool = False,
    ) -> HDMemoryTrace:
        """
        Encode text as a hypervector and store it.
        If a very similar trace already exists, reinforce it instead.
        """
        hv = self._codebook.text_to_hv(text)

        if not force_new:
            existing = self._find_similar_trace(hv, threshold=0.90)
            if existing:
                existing.fire()
                if metadata:
                    existing.metadata.update(metadata)
                if self.verbose:
                    logger.debug(f"[VSA] Reinforced: {existing.label}")
                return existing

        trace = HDMemoryTrace(
            id=str(uuid.uuid4()),
            label=label,
            hypervector=hv,
            metadata=metadata or {},
        )
        self._store[trace.id]      = trace
        self._text_cache[trace.id] = text
        self._store_count         += 1

        if topic:
            self._update_bundle(topic, hv, trace.weight)

        self._enforce_capacity()

        if self.verbose:
            logger.debug(f"[VSA] Stored: {label!r} (total={len(self._store)})")

        return trace

    def query(
        self,
        text: str,
        top_k: int = 5,
        topic: Optional[str] = None,
    ) -> List[QueryResult]:
        """Retrieve the top-k most similar memories to the query text."""
        if not self._store:
            return []

        self._query_count += 1
        query_hv = self._codebook.text_to_hv(text)

        if self.decay_enabled:
            self._run_decay()

        scored: List[tuple] = []
        for trace_id, trace in self._store.items():
            sim = _cosine_hd(query_hv, trace.hypervector)
            if sim >= self.similarity_threshold:
                scored.append((sim * trace.weight, sim, trace))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _weighted, sim, trace in scored[:top_k]:
            trace.fire()
            results.append(QueryResult(
                label=trace.label,
                text=self._text_cache.get(trace.id, ""),
                similarity=round(sim, 4),
                weight=round(trace.weight, 4),
                metadata=trace.metadata.copy(),
                trace_id=trace.id,
            ))

        if self.verbose:
            logger.debug(f"[VSA] Query: {len(results)}/{len(self._store)} results")

        return results

    def query_topic(self, topic: str, top_k: int = 5) -> List[QueryResult]:
        """Query using a pre-built topic bundle vector."""
        if topic not in self._bundle_cache:
            return []
        return self._query_hv(self._bundle_cache[topic], top_k=top_k)

    # ─────────────────────────────────────────────
    # Logical Operations
    # ─────────────────────────────────────────────

    def logical_and(self, text_a: str, text_b: str, top_k: int = 5) -> List[QueryResult]:
        """Find memories similar to BOTH text_a AND text_b."""
        hv_a = self._codebook.text_to_hv(text_a)
        hv_b = self._codebook.text_to_hv(text_b)
        return self._query_hv(_bundle([hv_a, hv_b]), top_k=top_k)

    def logical_not(self, text: str, top_k: int = 5) -> List[QueryResult]:
        """Find memories DISSIMILAR to text (semantic complement)."""
        hv = self._codebook.text_to_hv(text)
        return self._query_hv(1 - hv, top_k=top_k)

    # ─────────────────────────────────────────────
    # Memory Maintenance  (called by Aegis.sleep())
    # ─────────────────────────────────────────────

    def apply_temporal_decay(self) -> int:
        """
        Apply time-based weight decay to all stored traces.
        Called by Aegis.sleep() during end-of-session consolidation.

        Returns:
            Number of traces decayed.
        """
        return self._run_decay()

    def prune_weak(self) -> int:
        """
        Remove traces whose weight has fallen below prune_threshold,
        then enforce the capacity limit by removing the weakest survivors.
        Called by Aegis.sleep() during end-of-session consolidation.

        Returns:
            Number of traces removed.
        """
        before = len(self._store)

        weak_ids = [
            tid for tid, t in self._store.items()
            if t.weight < self.prune_threshold
        ]
        for tid in weak_ids:
            del self._store[tid]
            self._text_cache.pop(tid, None)

        self._enforce_capacity()

        pruned = before - len(self._store)
        if pruned and self.verbose:
            logger.debug(f"[VSA] Pruned {pruned} weak traces")
        return pruned

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _run_decay(self) -> int:
        """Decay all traces proportional to hours since last access."""
        now   = time.time()
        count = 0
        for trace in self._store.values():
            hours = (now - trace.last_accessed) / 3600.0
            if hours > 0.1:
                trace.decay(hours)
                count += 1
        return count

    def _enforce_capacity(self) -> None:
        """Remove weakest traces when store exceeds capacity."""
        if len(self._store) <= self.capacity:
            return
        sorted_traces = sorted(self._store.items(), key=lambda x: x[1].weight)
        overflow = len(self._store) - self.capacity
        for trace_id, _ in sorted_traces[:overflow]:
            del self._store[trace_id]
            self._text_cache.pop(trace_id, None)

    def _find_similar_trace(
        self, hv: np.ndarray, threshold: float
    ) -> Optional[HDMemoryTrace]:
        best_sim, best_trace = -1.0, None
        for trace in self._store.values():
            sim = _cosine_hd(hv, trace.hypervector)
            if sim > best_sim:
                best_sim, best_trace = sim, trace
        return best_trace if best_sim >= threshold else None

    def _update_bundle(self, topic: str, hv: np.ndarray, weight: float) -> None:
        if topic in self._bundle_cache:
            self._bundle_cache[topic] = _bundle(
                [self._bundle_cache[topic], hv], weights=[1.0, weight]
            )
        else:
            self._bundle_cache[topic] = hv.copy()

    def _query_hv(self, hv: np.ndarray, top_k: int = 5) -> List[QueryResult]:
        scored = []
        for trace_id, trace in self._store.items():
            sim = _cosine_hd(hv, trace.hypervector)
            if sim >= self.similarity_threshold:
                scored.append((sim * trace.weight, sim, trace))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            QueryResult(
                label=t.label,
                text=self._text_cache.get(t.id, ""),
                similarity=round(s, 4),
                weight=round(t.weight, 4),
                metadata=t.metadata.copy(),
                trace_id=t.id,
            )
            for _, s, t in scored[:top_k]
        ]

    # ─────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "stored_traces": len(self._store),
            "capacity":      self.capacity,
            "vocab_size":    self._codebook.vocab_size,
            "topics":        list(self._bundle_cache.keys()),
            "total_stores":  self._store_count,
            "total_queries": self._query_count,
            "dim":           self.dim,
        }

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"VSAMemory(dim={self.dim}, traces={len(self._store)}/{self.capacity})"