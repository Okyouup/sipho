"""
SynapticMemory: The persistent associative memory store.

Encodes experiences as weighted synaptic connections and retrieves
them by semantic similarity + synaptic strength.
"""

import json
import time
import uuid
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from Synapse import Synapse, _cosine_similarity
from Neurotrophic import NeurotrophicEngine


class SynapticMemory:
    """
    A brain-inspired associative memory system.

    Usage:
        memory = SynapticMemory(embed_fn=my_embedder)
        memory.encode("The sky is blue", metadata={"source": "observation"})
        results = memory.recall("What color is the sky?", top_k=3)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        persistence_path: Optional[str] = None,
        neurotrophic: Optional[NeurotrophicEngine] = None,
        recall_threshold: float = 0.25,
        auto_consolidate: bool = True,
    ):
        """
        Args:
            embed_fn: A function that takes a string and returns a float vector.
                      Can be any embedding model (OpenAI, sentence-transformers, etc.)
            persistence_path: Optional file path to save/load memory across sessions.
            neurotrophic: The neurotrophic engine for memory management.
            recall_threshold: Minimum relevance score to surface a memory.
            auto_consolidate: Run consolidation automatically on encode.
        """
        self.embed_fn = embed_fn
        self.persistence_path = persistence_path
        self.recall_threshold = recall_threshold
        self.auto_consolidate = auto_consolidate
        self.neurotrophic = neurotrophic or NeurotrophicEngine()
        self._synapses: Dict[str, Synapse] = {}
        self._encode_count = 0

        if persistence_path and os.path.exists(persistence_path):
            self.load(persistence_path)

    # ─────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────

    def encode(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Synapse]:
        """
        Encode a new memory trace. If a very similar memory exists,
        reinforce it instead of creating a duplicate (neuroplasticity).

        Args:
            text: The memory content to encode.
            metadata: Optional key-value context (source, timestamp, tags, etc.)
            force: If True, always create a new synapse regardless of similarity.

        Returns:
            The synapse that was created or reinforced.
        """
        vector = self.embed_fn(text)
        if not vector:
            return None

        # Check for existing similar memory
        if not force:
            existing = self._find_similar(vector, threshold=0.92)
            if existing:
                existing.fire()  # Reinforce existing memory
                existing.metadata.update(metadata or {})
                if self.persistence_path:
                    self.save(self.persistence_path)
                return existing

        # Create new synaptic connection
        synapse = Synapse(
            id=str(uuid.uuid4()),
            trace=text,
            context_vector=vector,
            metadata=metadata or {},
        )
        self._synapses[synapse.id] = synapse
        self._encode_count += 1

        # Periodic housekeeping
        if self.auto_consolidate and self._encode_count % 20 == 0:
            self.housekeeping()

        if self.persistence_path:
            self.save(self.persistence_path)

        return synapse

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_weight: float = 0.0,
        fire_on_recall: bool = True,
    ) -> List[Tuple[Synapse, float]]:
        """
        Retrieve memories most relevant to the query.
        Fires matching synapses (Hebbian reinforcement on recall).

        Returns:
            List of (Synapse, relevance_score) sorted by relevance descending.
        """
        if not self._synapses:
            return []

        query_vector = self.embed_fn(query)
        if not query_vector:
            return []

        scored = []
        for synapse in self._synapses.values():
            if synapse.weight < min_weight:
                continue
            score = synapse.relevance_score(query_vector)
            if score >= self.recall_threshold:
                scored.append((synapse, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        if fire_on_recall:
            for synapse, _ in results:
                synapse.fire()

        if self.persistence_path and results:
            self.save(self.persistence_path)

        return results

    def forget(self, synapse_id: str) -> bool:
        """Explicitly remove a memory."""
        if synapse_id in self._synapses:
            del self._synapses[synapse_id]
            return True
        return False

    def housekeeping(self) -> Dict[str, Any]:
        """
        Run full neurotrophic maintenance cycle:
        - Apply LTD decay
        - Prune weak synapses
        - Consolidate strong memories
        - Detect and merge duplicates
        """
        decayed = self.neurotrophic.apply_decay(self._synapses)
        pruned = self.neurotrophic.prune(self._synapses)
        consolidated = self.neurotrophic.consolidate(self._synapses)
        clusters = self.neurotrophic.detect_clusters(self._synapses)

        merged = []
        for id_a, id_b, sim in clusters[:5]:  # Merge top 5 clusters per cycle
            result = self.neurotrophic.merge_cluster(self._synapses, id_a, id_b)
            if result:
                merged.append(result.id)

        if self.persistence_path:
            self.save(self.persistence_path)

        return {
            "decayed": decayed,
            "pruned": len(pruned),
            "consolidated": len(consolidated),
            "merged": len(merged),
            "active_synapses": len(self._synapses),
        }

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize all synapses to JSON."""
        data = {
            "version": "1.0",
            "saved_at": time.time(),
            "synapses": {sid: s.to_dict() for sid, s in self._synapses.items()},
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> int:
        """Load synapses from JSON. Returns number of synapses loaded."""
        with open(path, "r") as f:
            data = json.load(f)
        self._synapses = {
            sid: Synapse.from_dict(s)
            for sid, s in data.get("synapses", {}).items()
        }
        return len(self._synapses)

    # ─────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────

    def _find_similar(self, vector: List[float], threshold: float) -> Optional[Synapse]:
        best, best_score = None, 0.0
        for synapse in self._synapses.values():
            score = _cosine_similarity(synapse.context_vector, vector)
            if score > best_score:
                best, best_score = synapse, score
        return best if best_score >= threshold else None

    @property
    def stats(self) -> dict:
        if not self._synapses:
            return {"total": 0}
        weights = [s.weight for s in self._synapses.values()]
        return {
            "total_synapses": len(self._synapses),
            "avg_weight": sum(weights) / len(weights),
            "max_weight": max(weights),
            "consolidated": sum(
                1 for s in self._synapses.values()
                if s.metadata.get("consolidated")
            ),
            "neurotrophic": self.neurotrophic.stats,
        }

    def __len__(self):
        return len(self._synapses)

    def __repr__(self):
        return f"SynapticMemory(synapses={len(self._synapses)}, threshold={self.recall_threshold})"