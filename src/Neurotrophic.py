"""
Neurotrophic Engine: Inspired by Brain-Derived Neurotrophic Factor (BDNF).

Responsibilities:
- Memory consolidation (short-term → long-term)
- Synaptic pruning (remove weak, redundant memories)
- Pattern detection (cluster frequently co-activated memories)
- Sleep-cycle consolidation (batch processing of memory traces)
"""

import time
import math
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from Synapse import Synapse, _cosine_similarity

if TYPE_CHECKING:
    pass

class NeurotrophicEngine:
    """
    The neurotrophic layer manages the health and evolution of synaptic memory.

    Key processes:
    1. **Consolidation**: Elevates frequently accessed short-term traces to
       long-term memory by boosting their synaptic weights.
    2. **Pruning**: Removes synapses that have decayed below the survival threshold.
    3. **Clustering**: Detects semantically similar memory traces and merges them
       to reduce redundancy (like hippocampal replay).
    4. **Plasticity scoring**: Rates how "learnable" a new memory is given context.
    """

    def __init__(
        self,
        pruning_threshold: float = 0.05,
        consolidation_threshold: int = 3,
        clustering_similarity: float = 0.92,
        max_synapses: int = 10_000,
    ):
        self.pruning_threshold = pruning_threshold          # Below this weight → pruned
        self.consolidation_threshold = consolidation_threshold  # Activations to consolidate
        self.clustering_similarity = clustering_similarity  # Similarity to merge
        self.max_synapses = max_synapses
        self._consolidation_log: List[Dict] = []
        self._pruning_log: List[Dict] = []

    def apply_decay(self, synapses: Dict[str, "Synapse"]) -> int:
        """
        Apply LTD decay to all synapses based on elapsed time since last firing.
        Returns the number of synapses decayed.
        """
        now = time.time()
        count = 0
        for synapse in synapses.values():
            hours_since_fire = (now - synapse.last_fired) / 3600.0
            if hours_since_fire > 0.1:  # Only decay if >6 minutes old
                synapse.decay(hours_since_fire)
                count += 1
        return count

    def prune(self, synapses: Dict[str, "Synapse"]) -> List[str]:
        """
        Remove weak synapses that have fallen below survival threshold.
        Returns list of pruned synapse IDs.
        """
        pruned = [
            sid for sid, s in synapses.items()
            if s.weight < self.pruning_threshold
        ]
        for sid in pruned:
            self._pruning_log.append({
                "synapse_id": sid,
                "trace_preview": synapses[sid].trace[:60],
                "final_weight": synapses[sid].weight,
                "pruned_at": time.time(),
            })
            del synapses[sid]

        # Also prune if over capacity — remove weakest
        if len(synapses) > self.max_synapses:
            sorted_by_weight = sorted(synapses.items(), key=lambda x: x[1].weight)
            overflow = len(synapses) - self.max_synapses
            for sid, s in sorted_by_weight[:overflow]:
                pruned.append(sid)
                del synapses[sid]

        return pruned

    def consolidate(self, synapses: Dict[str, "Synapse"]) -> List[str]:
        """
        Elevate frequently fired synapses to long-term memory by boosting weight
        and lowering their decay rate (making them more permanent).
        Returns IDs of consolidated synapses.
        """
        consolidated = []
        for sid, synapse in synapses.items():
            if (synapse.activations >= self.consolidation_threshold
                    and synapse.decay_rate > 0.001):
                # Long-term potentiation: reduce decay rate (memory becomes stable)
                synapse.decay_rate = max(synapse.decay_rate * 0.5, 0.001)
                synapse.weight = min(synapse.weight * 1.2, 10.0)
                synapse.metadata["consolidated"] = True
                synapse.metadata["consolidated_at"] = time.time()
                consolidated.append(sid)
                self._consolidation_log.append({
                    "synapse_id": sid,
                    "activations": synapse.activations,
                    "new_decay_rate": synapse.decay_rate,
                    "new_weight": synapse.weight,
                })
        return consolidated

    def detect_clusters(
        self, synapses: Dict[str, "Synapse"]
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of synapses that are semantically very similar.
        Returns list of (id_a, id_b, similarity) for near-duplicate memories.
        These can be merged or flagged for review.
        """

        synapse_list = list(synapses.items())
        clusters = []

        for i in range(len(synapse_list)):
            for j in range(i + 1, len(synapse_list)):
                sid_a, s_a = synapse_list[i]
                sid_b, s_b = synapse_list[j]
                if not s_a.context_vector or not s_b.context_vector:
                    continue
                sim = _cosine_similarity(s_a.context_vector, s_b.context_vector)
                if sim >= self.clustering_similarity:
                    clusters.append((sid_a, sid_b, sim))

        return clusters

    def merge_cluster(
        self,
        synapses: Dict[str, "Synapse"],
        id_a: str,
        id_b: str,
    ) -> Optional["Synapse"]:
        """
        Merge two redundant synapses into one stronger composite memory.
        The merged synapse inherits the combined strength of both.
        """
        if id_a not in synapses or id_b not in synapses:
            return None

        s_a, s_b = synapses[id_a], synapses[id_b]

        # Merged trace: prefer the heavier synapse's trace
        dominant = s_a if s_a.weight >= s_b.weight else s_b

        # Average the context vectors
        merged_vector = [
            (a + b) / 2.0
            for a, b in zip(s_a.context_vector, s_b.context_vector)
        ]

        merged = Synapse(
            id=str(uuid.uuid4()),
            trace=dominant.trace,
            context_vector=merged_vector,
            metadata={
                **dominant.metadata,
                "merged_from": [id_a, id_b],
                "merged_at": time.time(),
            },
            weight=min(s_a.weight + s_b.weight * 0.5, 10.0),
            activations=s_a.activations + s_b.activations,
            decay_rate=min(s_a.decay_rate, s_b.decay_rate),
        )

        del synapses[id_a]
        del synapses[id_b]
        synapses[merged.id] = merged
        return merged

    def plasticity_score(self, novelty: float, relevance: float) -> float:
        """
        Compute how readily a new memory should be encoded.
        High novelty + moderate relevance = high plasticity (brain encodes well).
        Low novelty (already known) = low plasticity (skip or reinforce existing).
        """
        return novelty * (1.0 - abs(relevance - 0.5) * 0.5)

    @property
    def stats(self) -> dict:
        return {
            "total_consolidated": len(self._consolidation_log),
            "total_pruned": len(self._pruning_log),
        }