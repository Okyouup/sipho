"""
Synapse: A weighted connection between a memory trace and its associated context.
Models Long-Term Potentiation (LTP) and Long-Term Depression (LTD).
"""

import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Synapse:
    """
    A single synaptic connection in the cognitive memory system.

    Inspired by biological synapses:
    - `weight` strengthens with use (LTP) and decays with disuse (LTD)
    - `trace` is the encoded memory content
    - `context` is the semantic vector fingerprint
    - `activations` tracks firing history
    """
    id: str
    trace: str                          # The stored memory / knowledge
    context_vector: List[float]         # Semantic embedding of the trace
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0                 # Synaptic strength (0.0 - ∞)
    activations: int = 0                # Times this synapse has fired
    created_at: float = field(default_factory=time.time)
    last_fired: float = field(default_factory=time.time)
    decay_rate: float = 0.01            # LTD decay per time unit
    potentiation_rate: float = 0.15     # LTP boost per activation

    def fire(self) -> float:
        """
        Activate this synapse. Strengthens weight (LTP).
        Returns the new synaptic weight.
        """
        self.activations += 1
        self.last_fired = time.time()
        # Hebbian potentiation — diminishing returns at high weights
        boost = self.potentiation_rate * math.exp(-self.weight * 0.1)
        self.weight = min(self.weight + boost, 10.0)
        return self.weight

    def decay(self, time_delta_hours: float = 1.0) -> float:
        """
        Apply LTD (synaptic depression) based on elapsed time.
        Unused synapses weaken over time.
        """
        decay_factor = math.exp(-self.decay_rate * time_delta_hours)
        self.weight = max(self.weight * decay_factor, 0.01)
        return self.weight

    def relevance_score(self, query_vector: List[float]) -> float:
        """
        Compute relevance as: cosine_similarity × synaptic_weight.
        Strong + relevant memories surface first.
        """
        cos_sim = _cosine_similarity(self.context_vector, query_vector)
        return cos_sim * self.weight

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trace": self.trace,
            "context_vector": self.context_vector,
            "metadata": self.metadata,
            "weight": self.weight,
            "activations": self.activations,
            "created_at": self.created_at,
            "last_fired": self.last_fired,
            "decay_rate": self.decay_rate,
            "potentiation_rate": self.potentiation_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Synapse":
        return cls(**data)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)