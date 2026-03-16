"""
PerceptualGateway: SNN-based Perceptual Router (Phase 1 of Aegis-1).

Uses a Nengo Leaky Integrate-and-Fire network to assess input novelty
and route queries to System 1 (fast/cheap) or System 2 (deliberate/expensive).

Inspired by the thalamo-cortical gating model: the thalamus acts as a relay
that can amplify or suppress signals before they reach the cortex.

v2 improvement:
- When Nengo is unavailable, replaces the naive sigmoid fallback with an
  analytical LIF rate-model that mirrors the actual network topology:
  excitatory drive → LIF firing → feed-forward inhibition → lateral feedback.
  Routing behaviour is now consistent whether or not Nengo is installed.
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
    SYSTEM_1 = "system_1"
    SYSTEM_2 = "system_2"


@dataclass
class GatewayDecision:
    route: Route
    novelty: float
    spike_rate: float
    confidence: float
    surprisal: float
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
# SNN Implementations
# ─────────────────────────────────────────────────────────────────────────────

def _lif_rate(
    J: float,
    tau_rc: float   = 0.020,
    tau_ref: float  = 0.002,
    max_rate: float = 200.0,
) -> float:
    """
    Analytical mean firing rate of a Leaky Integrate-and-Fire neuron.

    r = 1 / (tau_ref - tau_rc * ln(1 - 1/J))  for J > 1
    r = 0                                        for J <= 1

    Returns normalised rate in [0, 1].
    """
    if J <= 1.0:
        return 0.0
    try:
        denom = tau_ref + tau_rc * math.log(1.0 - 1.0 / J)
        if denom <= 0:
            return 1.0
        return min(1.0 / denom / max_rate, 1.0)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _snn_analytical_fallback(novelty_signal: float) -> float:
    """
    Analytical approximation of the Nengo LIF network with lateral inhibition.

    Mirrors the three-layer topology of _build_snn_network:
      1. Excitatory drive  → LIF rate (J scaled to [1, 4])
      2. Feed-forward      → inhibitory rate  (weight 0.6)
      3. Lateral feedback  → corrected excitatory rate  (weight -0.4)

    Produces a sharper threshold and stronger sub-threshold suppression
    than a plain sigmoid, matching the Nengo network's behaviour.
    """
    J_exc = 1.0 + novelty_signal * 3.0
    r_exc = _lif_rate(J_exc, tau_rc=0.020, tau_ref=0.002, max_rate=200.0)

    J_inh = 1.0 + r_exc * 0.6 * 3.0
    r_inh = _lif_rate(J_inh, tau_rc=0.010, tau_ref=0.001, max_rate=250.0)

    J_exc_corrected = max(J_exc - 0.4 * r_inh * 3.0, 0.0)
    r_final = _lif_rate(J_exc_corrected, tau_rc=0.020, tau_ref=0.002, max_rate=200.0)

    return float(np.clip(r_final, 0.0, 1.0))


def _build_snn_network(
    novelty_signal: float,
    n_excitatory: int   = 120,
    n_inhibitory: int   = 60,
    sim_duration: float = 0.15,
    dt: float           = 0.001,
) -> float:
    """
    Run the Nengo LIF network (if available) or the analytical fallback.

    Returns normalised mean firing rate in [0, 1].
    """
    if not _NENGO_AVAILABLE:
        return _snn_analytical_fallback(novelty_signal)

    captured = [novelty_signal]

    with nengo.Network(seed=42) as net:
        stim = nengo.Node(output=lambda t: captured[0])

        excitatory = nengo.Ensemble(
            n_neurons=n_excitatory, dimensions=1,
            neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
            max_rates=nengo.dists.Uniform(100, 200),
            intercepts=nengo.dists.Uniform(-0.5, 0.5),
        )
        inhibitory = nengo.Ensemble(
            n_neurons=n_inhibitory, dimensions=1,
            neuron_type=nengo.LIF(tau_rc=0.01, tau_ref=0.001),
            max_rates=nengo.dists.Uniform(150, 250),
        )

        nengo.Connection(stim,       excitatory, synapse=0.005)
        nengo.Connection(excitatory, inhibitory, synapse=0.005, transform=0.6)
        nengo.Connection(inhibitory, excitatory, synapse=0.005, transform=-0.4)

        p_excitatory = nengo.Probe(excitatory, synapse=0.01)

    with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
        sim.run(sim_duration)

    steady_start = int(len(sim.data[p_excitatory]) * 0.66)
    mean_output  = float(np.mean(sim.data[p_excitatory][steady_start:]))
    return float(np.clip((mean_output + 1.0) / 2.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual Gateway
# ─────────────────────────────────────────────────────────────────────────────

class PerceptualGateway:
    """
    Thalamic gate for Aegis-1.

    Analyzes each user input and decides whether it warrants full System 2
    reasoning or can be handled by the fast System 1 path.
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
        self.spike_threshold        = spike_threshold
        self.embed_fn               = embed_fn
        self.snn_excitatory_neurons = snn_excitatory_neurons
        self.snn_inhibitory_neurons = snn_inhibitory_neurons
        self.snn_sim_duration       = snn_sim_duration
        self.verbose                = verbose

        self._history: deque        = deque(maxlen=novelty_window)
        self._token_freq: Dict[str, int] = {}
        self._total_tokens: int     = 0
        self._s1_count: int         = 0
        self._s2_count: int         = 0
        self._latency_history: List[float] = []

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def assess(self, text: str) -> GatewayDecision:
        t0 = time.perf_counter()

        surprisal = self._compute_surprisal(text)
        novelty   = self._compute_novelty(text)
        composite = float(np.clip(0.4 * surprisal + 0.6 * novelty, 0.0, 1.0))

        spike_rate = _build_snn_network(
            novelty_signal=composite,
            n_excitatory=self.snn_excitatory_neurons,
            n_inhibitory=self.snn_inhibitory_neurons,
            sim_duration=self.snn_sim_duration,
        )

        route      = Route.SYSTEM_2 if spike_rate >= self.spike_threshold else Route.SYSTEM_1
        confidence = float(np.clip(
            abs(spike_rate - self.spike_threshold) / max(self.spike_threshold, 1e-6),
            0.0, 1.0,
        ))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latency_history.append(latency_ms)
        self._update_baseline(text)

        if route == Route.SYSTEM_1:
            self._s1_count += 1
        else:
            self._s2_count += 1

        decision = GatewayDecision(
            route=route, novelty=novelty, spike_rate=spike_rate,
            confidence=confidence, surprisal=surprisal, latency_ms=latency_ms,
            metadata={
                "composite_signal": composite,
                "nengo_available":  _NENGO_AVAILABLE,
                "snn_mode":         "nengo" if _NENGO_AVAILABLE else "analytical_lif",
            },
        )

        if self.verbose:
            logger.info(f"[Gateway] {decision}")

        return decision

    def calibrate_threshold(self, labeled_examples: List[tuple]) -> float:
        s1_rates, s2_rates = [], []
        for text, label in labeled_examples:
            d = self.assess(text)
            (s1_rates if label == Route.SYSTEM_1 else s2_rates).append(d.spike_rate)
        if s1_rates and s2_rates:
            self.spike_threshold = (
                sum(s1_rates) / len(s1_rates) + sum(s2_rates) / len(s2_rates)
            ) / 2.0
        return self.spike_threshold

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _compute_surprisal(self, text: str) -> float:
        if not text:
            return 0.0
        counts: Dict[str, int] = {}
        for ch in text.lower():
            counts[ch] = counts.get(ch, 0) + 1
        n = len(text)
        entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
        return float(np.clip(entropy / 6.57, 0.0, 1.0))

    def _compute_novelty(self, text: str) -> float:
        if self.embed_fn and self._history:
            try:
                vec = np.array(self.embed_fn(text), dtype=np.float32)
                history_vecs = [
                    np.array(self.embed_fn(h), dtype=np.float32)
                    for h in list(self._history)[-5:]
                ]
                similarities = [
                    float(np.dot(vec, h) / (np.linalg.norm(vec) * np.linalg.norm(h) + 1e-8))
                    for h in history_vecs
                ]
                return float(np.clip(1.0 - max(similarities), 0.0, 1.0))
            except Exception:
                pass

        tokens = text.lower().split()
        if not tokens or self._total_tokens == 0:
            return 1.0
        scores = []
        for tok in tokens:
            freq = self._token_freq.get(tok, 0)
            p    = (freq + 1) / (self._total_tokens + len(self._token_freq) + 1)
            scores.append(-math.log2(p))
        return float(np.clip(sum(scores) / len(scores) / 20.0, 0.0, 1.0))

    def _update_baseline(self, text: str) -> None:
        self._history.append(text)
        for tok in text.lower().split():
            self._token_freq[tok]  = self._token_freq.get(tok, 0) + 1
            self._total_tokens    += 1

    @property
    def stats(self) -> Dict:
        total   = self._s1_count + self._s2_count
        avg_lat = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history else 0.0
        )
        return {
            "total_assessments": total,
            "system_1_count":    self._s1_count,
            "system_2_count":    self._s2_count,
            "s2_ratio":          self._s2_count / total if total else 0.0,
            "spike_threshold":   self.spike_threshold,
            "avg_latency_ms":    round(avg_lat, 2),
            "nengo_available":   _NENGO_AVAILABLE,
            "snn_mode":          "nengo" if _NENGO_AVAILABLE else "analytical_lif",
        }