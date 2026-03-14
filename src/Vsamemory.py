"""
VSAMemory: Hyperdimensional Computing (HDC) Memory Kernel (Phase 2 of Aegis-1).

Unlike vector databases that perform approximate nearest-neighbour *search*,
VSA is *associative*: concepts are bound into composite hypervectors that
support logical operations (AND, OR, NOT equivalents) before retrieval.

Theory:
    - Binary hypervectors in R^D (D = 10,000 by default)
    - Binding   : XOR          (associates two concepts; recoverable)
    - Bundling  : majority vote (creates a superposition / "OR" memory)
    - Similarity: normalised Hamming distance

Reference:
    Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to
    Computing in Distributed Representation with High-Dimensional Random Vectors.

Requirements:
    pip install numpy
"""

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HDMemoryTrace:
    """A single entry in the HDC memory store."""
    id: str
    label: str                          # Human-readable key
    hypervector: np.ndarray             # Binary {0, 1}^D
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0                 # Salience (mirrors Synapse weight)
    activations: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.005

    def fire(self) -> None:
        self.activations += 1
        self.last_accessed = time.time()
        boost = 0.1 * math.exp(-self.weight * 0.1)
        self.weight = min(self.weight + boost, 10.0)

    def decay(self, hours_elapsed: float) -> None:
        factor = math.exp(-self.decay_rate * hours_elapsed)
        self.weight = max(self.weight * factor, 0.01)


@dataclass
class QueryResult:
    """Result from a VSA memory query."""
    label: str
    similarity: float           # Cosine similarity in HD space [0, 1]
    weight: float
    score: float                # similarity × weight (combined salience)
    metadata: Dict[str, Any]
    trace_id: str


# ─────────────────────────────────────────────────────────────────────────────
# VSA Operations (static)
# ─────────────────────────────────────────────────────────────────────────────

def _random_hv(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random binary hypervector {0, 1}^D with ~50% density."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def _bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Binding via XOR — associates two hypervectors into a composite.
    Recoverable: bind(bind(a, b), b) ≈ a
    """
    return np.bitwise_xor(a, b)


def _bundle(vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Bundle (superposition) via weighted majority vote.
    The result is approximately similar to all inputs.

    Args:
        vectors: List of binary hypervectors.
        weights: Optional salience weights per vector.

    Returns:
        A binary hypervector representing the bundle.
    """
    if not vectors:
        raise ValueError("Cannot bundle empty list.")
    if len(vectors) == 1:
        return vectors[0].copy()

    weights = weights or [1.0] * len(vectors)
    stack = np.stack(vectors, axis=0).astype(np.float32)
    w = np.array(weights, dtype=np.float32)[:, np.newaxis]
    weighted_sum = (stack * w).sum(axis=0)
    threshold = w.sum() / 2.0
    return (weighted_sum >= threshold).astype(np.uint8)


def _cosine_hd(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two binary hypervectors.
    Equivalent to 1 - normalised Hamming distance for balanced vectors.
    """
    a_f = a.astype(np.float32) * 2 - 1  # Map {0,1} → {-1,+1}
    b_f = b.astype(np.float32) * 2 - 1
    dot = float(np.dot(a_f, b_f))
    norm = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    return dot / norm if norm > 0 else 0.0


def _permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Cyclic permutation — encodes sequential position in VSA sequences."""
    return np.roll(v, shift)


# ─────────────────────────────────────────────────────────────────────────────
# Codebook: maps tokens/concepts to fixed random hypervectors
# ─────────────────────────────────────────────────────────────────────────────

class HDCodebook:
    """
    Maps string tokens to stable random hypervectors.
    Once a token is assigned a hypervector, it never changes.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._book: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed=2024)

    def encode(self, token: str) -> np.ndarray:
        """Return (or create) the base hypervector for a token."""
        if token not in self._book:
            hv = self._rng.integers(0, 2, size=self.dim, dtype=np.uint8)
            self._book[token] = hv
        return self._book[token]

    def encode_sequence(self, tokens: List[str]) -> np.ndarray:
        """
        Encode an ordered sequence via position-shifted binding + bundling.
        Each token's hypervector is permuted by its position index.
        """
        components = [
            _permute(self.encode(tok), shift=i)
            for i, tok in enumerate(tokens)
        ]
        return _bundle(components)

    def text_to_hv(self, text: str) -> np.ndarray:
        """Convert raw text to a hypervector via n-gram tokenisation."""
        tokens = self._tokenise(text)
        if not tokens:
            return _random_hv(self.dim)
        # Unigrams
        unigram_hvs = [self.encode(t) for t in tokens]
        # Bigrams (position-aware binding)
        bigram_hvs = [
            _bind(self.encode(tokens[i]), _permute(self.encode(tokens[i + 1])))
            for i in range(len(tokens) - 1)
        ]
        all_hvs = unigram_hvs + bigram_hvs
        return _bundle(all_hvs)

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
        verbose: bool = False,
    ):
        """
        Args:
            dim: Hypervector dimensionality. 10k is standard HDC practice.
            capacity: Max number of stored traces (prune weakest when exceeded).
            similarity_threshold: Minimum cosine similarity for retrieval.
            decay_enabled: Apply temporal weight decay.
            verbose: Log operations.
        """
        self.dim = dim
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.decay_enabled = decay_enabled
        self.verbose = verbose

        self._codebook = HDCodebook(dim=dim)
        self._store: Dict[str, HDMemoryTrace] = {}
        self._bundle_cache: Dict[str, np.ndarray] = {}  # topic → bundled HV

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

        If a very similar trace already exists, reinforce it (HDC plasticity).

        Args:
            label: Human-readable key.
            text: The knowledge / memory content.
            metadata: Optional context dict.
            topic: Optional topic group for bundling.
            force_new: Always create a new entry.

        Returns:
            The stored or reinforced HDMemoryTrace.
        """
        hv = self._codebook.text_to_hv(text)

        if not force_new:
            existing = self._find_similar_trace(hv, threshold=0.90)
            if existing:
                existing.fire()
                if metadata:
                    existing.metadata.update(metadata)
                if self.verbose:
                    logger.debug(f"[VSA] Reinforced existing trace: {existing.label}")
                return existing

        trace = HDMemoryTrace(
            id=str(uuid.uuid4()),
            label=label,
            hypervector=hv,
            metadata=metadata or {},
        )
        self._store[trace.id] = trace

        # Update topic bundle
        if topic:
            self._update_bundle(topic, hv, trace.weight)

        self._enforce_capacity()

        if self.verbose:
            logger.debug(f"[VSA] Stored: {label} | dim={self.dim} | total={len(self._store)}")

        return trace

    def query(
        self,
        text: str,
        top_k: int = 5,
        topic_filter: Optional[str] = None,
        fire_on_recall: bool = True,
    ) -> List[QueryResult]:
        """
        Retrieve memories associatively similar to the query.

        Args:
            text: Query string.
            top_k: Maximum results.
            topic_filter: Optionally restrict to a topic bundle.
            fire_on_recall: Reinforce retrieved traces (Hebbian learning).

        Returns:
            List of QueryResult sorted by score descending.
        """
        if not self._store:
            return []

        query_hv = self._codebook.text_to_hv(text)

        # Optionally mask with topic bundle (logical AND via weighted query)
        if topic_filter and topic_filter in self._bundle_cache:
            bundle_hv = self._bundle_cache[topic_filter]
            # Modulate query by topic: soft AND via blended hypervector
            query_hv = _bundle([query_hv, bundle_hv], weights=[0.7, 0.3])

        scored: List[QueryResult] = []
        for trace in self._store.values():
            sim = _cosine_hd(trace.hypervector, query_hv)
            if sim >= self.similarity_threshold:
                score = sim * trace.weight
                scored.append(QueryResult(
                    label=trace.label,
                    similarity=sim,
                    weight=trace.weight,
                    score=score,
                    metadata=trace.metadata,
                    trace_id=trace.id,
                ))

        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:top_k]

        if fire_on_recall:
            for r in results:
                self._store[r.trace_id].fire()

        return results

    def bind_concepts(
        self,
        concept_a: str,
        concept_b: str,
        label: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> HDMemoryTrace:
        """
        Create a bound association between two concepts and store it.
        Example: bind_concepts("Paris", "capital of France")
        """
        hv_a = self._codebook.text_to_hv(concept_a)
        hv_b = self._codebook.text_to_hv(concept_b)
        bound = _bind(hv_a, hv_b)

        lbl = label or f"{concept_a}::{concept_b}"
        trace = HDMemoryTrace(
            id=str(uuid.uuid4()),
            label=lbl,
            hypervector=bound,
            metadata=metadata or {"concept_a": concept_a, "concept_b": concept_b},
        )
        self._store[trace.id] = trace
        return trace

    def logical_query(
        self,
        include: List[str],
        exclude: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[QueryResult]:
        """
        Logical compound query:
            include → bundled (OR) query vector
            exclude → subtracted from query (NOT approximation)

        Args:
            include: Concepts to include (OR).
            exclude: Concepts to negate (NOT).
            top_k: Max results.
        """
        include_hvs = [self._codebook.text_to_hv(t) for t in include]
        query_hv = _bundle(include_hvs) if len(include_hvs) > 1 else include_hvs[0]

        if exclude:
            exclude_hvs = [self._codebook.text_to_hv(t) for t in exclude]
            exclude_bundle = _bundle(exclude_hvs) if len(exclude_hvs) > 1 else exclude_hvs[0]
            # NOT approximation: XOR with exclusion bundle (flips relevant bits)
            query_hv = _bind(query_hv, exclude_bundle)

        scored: List[QueryResult] = []
        for trace in self._store.values():
            sim = _cosine_hd(trace.hypervector, query_hv)
            if exclude:
                # Penalise traces similar to excluded concepts
                for ehv in [self._codebook.text_to_hv(e) for e in (exclude or [])]:
                    penalty = max(0.0, _cosine_hd(trace.hypervector, ehv))
                    sim = sim - 0.5 * penalty
            if sim >= self.similarity_threshold * 0.5:
                scored.append(QueryResult(
                    label=trace.label,
                    similarity=sim,
                    weight=trace.weight,
                    score=sim * trace.weight,
                    metadata=trace.metadata,
                    trace_id=trace.id,
                ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def apply_temporal_decay(self) -> int:
        """
        Decay all traces based on time since last access.
        Returns number of traces decayed.
        """
        if not self.decay_enabled:
            return 0
        now = time.time()
        count = 0
        for trace in list(self._store.values()):
            hours = (now - trace.last_accessed) / 3600.0
            if hours > 0.1:
                trace.decay(hours)
                count += 1
        return count

    def prune_weak(self, weight_threshold: float = 0.05) -> int:
        """Remove traces below weight threshold. Returns count pruned."""
        to_prune = [
            tid for tid, t in self._store.items()
            if t.weight < weight_threshold
        ]
        for tid in to_prune:
            del self._store[tid]
        return len(to_prune)

    def forget(self, trace_id: str) -> bool:
        """Explicitly remove a trace."""
        if trace_id in self._store:
            del self._store[trace_id]
            return True
        return False

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _find_similar_trace(
        self, hv: np.ndarray, threshold: float
    ) -> Optional[HDMemoryTrace]:
        best, best_sim = None, 0.0
        for trace in self._store.values():
            sim = _cosine_hd(trace.hypervector, hv)
            if sim > best_sim:
                best, best_sim = trace, sim
        return best if best_sim >= threshold else None

    def _update_bundle(self, topic: str, hv: np.ndarray, weight: float) -> None:
        if topic not in self._bundle_cache:
            self._bundle_cache[topic] = hv.copy()
        else:
            existing = self._bundle_cache[topic]
            self._bundle_cache[topic] = _bundle([existing, hv], weights=[1.0, weight])

    def _enforce_capacity(self) -> None:
        if len(self._store) > self.capacity:
            sorted_traces = sorted(self._store.items(), key=lambda x: x[1].weight)
            overflow = len(self._store) - self.capacity
            for tid, _ in sorted_traces[:overflow]:
                del self._store[tid]

    @property
    def stats(self) -> Dict:
        if not self._store:
            return {"total": 0, "dim": self.dim}
        weights = [t.weight for t in self._store.values()]
        return {
            "total_traces": len(self._store),
            "dim": self.dim,
            "vocab_size": self._codebook.vocab_size,
            "avg_weight": round(sum(weights) / len(weights), 4),
            "max_weight": round(max(weights), 4),
            "topic_bundles": list(self._bundle_cache.keys()),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"VSAMemory(traces={len(self._store)}, dim={self.dim})"