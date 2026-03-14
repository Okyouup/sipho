"""
PerceptualGateway: SNN-based Perceptual Router (Phase 1 of Aegis-1).

Uses a Nengo Leaky Integrate-and-Fire network to assess input novelty
and route queries to System 1 (fast/cheap) or System 2 (deliberate/expensive).

Inspired by the thalamo-cortical gating model: the thalamus acts as a relay
that can amplify or suppress signals before they reach the cortex.

"""

import math
import time
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import nengo
    import nengo.utils.ensemble
    _NENGO_AVAILABLE = True
except ImportError:
    _NENGO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class Route(Enum):
    SYSTEM_1 = "system_1"   # Fast, cheap: LLM with minimal context
    SYSTEM_2 = "system_2"   # Deliberate: full cognitive loop with VSA + monitor


@dataclass
class GatewayDecision:
    """Result of the perceptual assessment."""
    route: Route
    novelty: float          # 0.0 (fully familiar) → 1.0 (completely novel)
    spike_rate: float       # Normalised SNN output firing rate
    confidence: float       # How certain the routing decision is
    surprisal: float        # Per-character entropy of the input
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)

    @property
    def requires_deep_reasoning(self) -> bool:
        return self.route == Route.SYSTEM_2

    def __repr__(self) -> str:
        return (
            f"GatewayDecision(route={self.route.value}, "
            f"novelty={self.novelty:.3f}, spike_rate={self.spike_rate:.3f}, "
            f"confidence={self.confidence:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Nengo SNN Network Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_snn_network(
    novelty_signal: float,
    n_excitatory: int = 120,
    n_inhibitory: int = 60,
    sim_duration: float = 0.15,
    dt: float = 0.001,
) -> float:
    """
    Build and run a single-pass Nengo LIF network.

    Topology:
        Input → Excitatory Ensemble ←→ Inhibitory Ensemble (lateral inhibition)
                        ↓
                   Decision Probe

    The excitatory population fires proportionally to novelty.
    The inhibitory interneurons suppress weak signals (thresholding).

    Returns:
        Normalised mean firing rate from excitatory population [0, 1].
    """
    if not _NENGO_AVAILABLE:
        # Analytic fallback: sigmoid activation
        return 1.0 / (1.0 + math.exp(-10.0 * (novelty_signal - 0.4)))

    captured_signal = [novelty_signal]  # closure capture

    with nengo.Network(seed=42) as net:
        # ── Input ──
        stim = nengo.Node(output=lambda t: captured_signal[0])

        # ── Excitatory population (LIF) ──
        excitatory = nengo.Ensemble(
            n_neurons=n_excitatory,
            dimensions=1,
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=nengo.dists.Uniform(-0.5, 0.5),
        )

        # ── Inhibitory interneurons (LIF, faster time constants) ──
        inhibitory = nengo.Ensemble(
            n_neurons=n_inhibitory,
            dimensions=1,
            neuron_type=nengo.LIF(tau_rc=0.01, tau_ref=0.001),
            max_rates=nengo.dists.Uniform(150, 250),
        )

        # ── Connections ──
        nengo.Connection(stim, excitatory, synapse=0.005)
        # Feed-forward excitation to inhibitory layer
        nengo.Connection(excitatory, inhibitory, synapse=0.005, transform=0.6)
        # Lateral inhibition back onto excitatory (sharpens threshold)
        nengo.Connection(inhibitory, excitatory, synapse=0.005, transform=-0.4)

        # ── Probes ──
        p_excitatory = nengo.Probe(excitatory, synapse=0.01)

    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(sim_duration)

    # Mean decoded value over the last third of simulation (steady state)
    steady_start = int(len(sim.data[p_excitatory]) * 0.66)
    mean_output = float(np.mean(sim.data[p_excitatory][steady_start:]))

    # Normalise: decoded value is in [-1, 1] approximately
    normalised = (mean_output + 1.0) / 2.0
    return float(np.clip(normalised, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Gateway
# ─────────────────────────────────────────────────────────────────────────────

class PerceptualGateway:
    """
    Thalamic gate for Aegis-1.

    Analyzes each user input and decides whether it warrants full System 2
    reasoning or can be handled by the fast System 1 path.

    Usage:
        gateway = PerceptualGateway(spike_threshold=0.55)
        decision = gateway.assess("What is the capital of France?")
        if decision.requires_deep_reasoning:
            # invoke full Aegis cognitive loop
        else:
            # fast LLM pass-through
    """

    def __init__(
        self,
        spike_threshold: float = 0.55,
        novelty_window: int = 50,
        s1_label: str = "SYSTEM_1",
        s2_label: str = "SYSTEM_2",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        snn_excitatory_neurons: int = 120,
        snn_inhibitory_neurons: int = 60,
        snn_sim_duration: float = 0.10,
        verbose: bool = False,
    ):
        """
        Args:
            spike_threshold: SNN firing rate above which System 2 is triggered.
            novelty_window: How many past inputs to track for baseline novelty.
            embed_fn: Optional embedder for semantic novelty scoring.
                      Falls back to character n-gram entropy if None.
            snn_excitatory_neurons: LIF neurons in excitatory population.
            snn_inhibitory_neurons: LIF neurons in inhibitory population.
            snn_sim_duration: Simulated time per assessment (seconds).
            verbose: Log gateway decisions.
        """
        self.spike_threshold = spike_threshold
        self.embed_fn = embed_fn
        self.snn_excitatory_neurons = snn_excitatory_neurons
        self.snn_inhibitory_neurons = snn_inhibitory_neurons
        self.snn_sim_duration = snn_sim_duration
        self.verbose = verbose

        # Rolling window of recent input fingerprints for novelty baseline
        self._history: deque = deque(maxlen=novelty_window)
        self._token_freq: Dict[str, int] = {}
        self._total_tokens: int = 0

        # Metrics
        self._s1_count: int = 0
        self._s2_count: int = 0
        self._latency_history: List[float] = []

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def assess(self, text: str) -> GatewayDecision:
        """
        Assess the novelty of an input and decide routing.

        Args:
            text: The raw user input string.

        Returns:
            GatewayDecision with routing recommendation.
        """
        t0 = time.perf_counter()

        # Step 1: Compute surprisal (character-level entropy)
        surprisal = self._compute_surprisal(text)

        # Step 2: Semantic novelty vs recent context
        novelty = self._compute_novelty(text)

        # Step 3: Composite signal (blend surprisal + novelty)
        composite = 0.4 * surprisal + 0.6 * novelty
        composite = float(np.clip(composite, 0.0, 1.0))

        # Step 4: Run SNN to get spike rate
        spike_rate = _build_snn_network(
            novelty_signal=composite,
            n_excitatory=self.snn_excitatory_neurons,
            n_inhibitory=self.snn_inhibitory_neurons,
            sim_duration=self.snn_sim_duration,
        )

        # Step 5: Route decision
        route = Route.SYSTEM_2 if spike_rate >= self.spike_threshold else Route.SYSTEM_1

        # Step 6: Confidence (distance from threshold)
        confidence = abs(spike_rate - self.spike_threshold) / max(self.spike_threshold, 1e-6)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_history.append(latency_ms)

        # Update history
        self._update_baseline(text)
        if route == Route.SYSTEM_1:
            self._s1_count += 1
        else:
            self._s2_count += 1

        decision = GatewayDecision(
            route=route,
            novelty=novelty,
            spike_rate=spike_rate,
            confidence=confidence,
            surprisal=surprisal,
            latency_ms=latency_ms,
            metadata={"composite_signal": composite, "nengo_available": _NENGO_AVAILABLE},
        )

        if self.verbose:
            logger.info(f"[Gateway] {decision}")

        return decision

    def calibrate_threshold(self, labeled_examples: List[tuple]) -> float:
        """
        Tune spike_threshold given labelled (text, Route) examples.
        Uses simple bisection on mean spike rates per class.

        Args:
            labeled_examples: List of (text, Route) pairs.

        Returns:
            The calibrated threshold (also sets self.spike_threshold).
        """
        s1_rates, s2_rates = [], []
        for text, label in labeled_examples:
            decision = self.assess(text)
            if label == Route.SYSTEM_1:
                s1_rates.append(decision.spike_rate)
            else:
                s2_rates.append(decision.spike_rate)

        if s1_rates and s2_rates:
            self.spike_threshold = (
                sum(s1_rates) / len(s1_rates) + sum(s2_rates) / len(s2_rates)
            ) / 2.0

        return self.spike_threshold

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _compute_surprisal(self, text: str) -> float:
        """
        Estimate per-character entropy (surprisal) of the text.
        High entropy → unusual / complex → higher surprisal.
        """
        if not text:
            return 0.0
        char_counts: Dict[str, int] = {}
        for ch in text.lower():
            char_counts[ch] = char_counts.get(ch, 0) + 1
        n = len(text)
        entropy = -sum(
            (c / n) * math.log2(c / n)
            for c in char_counts.values() if c > 0
        )
        # Normalise: max entropy for ASCII is ~log2(95) ≈ 6.57
        return float(np.clip(entropy / 6.57, 0.0, 1.0))

    def _compute_novelty(self, text: str) -> float:
        """
        Semantic novelty: how different is this input from recent history?

        If embed_fn is available, uses cosine distance.
        Otherwise falls back to token-level inverse frequency.
        """
        if self.embed_fn and self._history:
            try:
                vec = np.array(self.embed_fn(text), dtype=np.float32)
                history_vecs = [
                    np.array(self.embed_fn(h), dtype=np.float32)
                    for h in list(self._history)[-5:]  # last 5 inputs
                ]
                similarities = [
                    float(np.dot(vec, h) / (np.linalg.norm(vec) * np.linalg.norm(h) + 1e-8))
                    for h in history_vecs
                ]
                max_sim = max(similarities) if similarities else 0.0
                return float(np.clip(1.0 - max_sim, 0.0, 1.0))
            except Exception:
                pass  # graceful fallback

        # Token frequency novelty: rare tokens → high novelty
        tokens = text.lower().split()
        if not tokens or self._total_tokens == 0:
            return 1.0

        scores = []
        for tok in tokens:
            freq = self._token_freq.get(tok, 0)
            p = (freq + 1) / (self._total_tokens + len(self._token_freq) + 1)
            scores.append(-math.log2(p))  # surprisal in bits

        # Normalise by a rough maximum surprisal (20 bits ≈ 1-in-1M token)
        mean_surprisal = sum(scores) / len(scores)
        return float(np.clip(mean_surprisal / 20.0, 0.0, 1.0))

    def _update_baseline(self, text: str) -> None:
        """Update token frequency model and history buffer."""
        self._history.append(text)
        for tok in text.lower().split():
            self._token_freq[tok] = self._token_freq.get(tok, 0) + 1
            self._total_tokens += 1

    @property
    def stats(self) -> Dict:
        total = self._s1_count + self._s2_count
        avg_lat = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history else 0.0
        )
        return {
            "total_assessments": total,
            "system_1_count": self._s1_count,
            "system_2_count": self._s2_count,
            "s2_ratio": self._s2_count / total if total else 0.0,
            "spike_threshold": self.spike_threshold,
            "avg_latency_ms": round(avg_lat, 2),
            "nengo_available": _NENGO_AVAILABLE,
        }