import re
import time
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class QualityFlag(Enum):
    UNCERTAIN       = "uncertain"       # Response hedges heavily
    SHALLOW         = "shallow"         # Response is too brief/vague
    VERBOSE         = "verbose"         # Response is padded / over-long
    CONTRADICTORY   = "contradictory"   # Self-contradiction detected
    OVER_HEDGED     = "over_hedged"     # Excessive uncertainty qualifiers
    UNGROUNDED      = "ungrounded"      # No retrieved memories supported it
    CONFLICT_FAILED = "conflict_failed" # Monitor flagged conflicts
    HIGH_CONFIDENCE = "high_confidence" # Response looks solid
    LOW_MEMORY_USE  = "low_memory_use"  # Few memories retrieved (possible gap)


@dataclass
class MetaCognitionReport:
    """Self-evaluation output for a single LLM response."""
    confidence: float               # [0, 1] — 0 = unreliable, 1 = certain
    reasoning_quality: float        # [0, 1] — structural quality of reasoning
    flags: List[QualityFlag]        # Detected quality issues
    hedge_density: float            # Fraction of hedging phrases
    response_length: int            # Word count of response
    memory_hits: int                # Memories retrieved during this turn
    conflict_count: int             # Conflicts detected by ExecutiveMonitor
    should_rethink: bool            # Confidence so low we should retry
    annotation: str                 # Human-readable confidence label
    latency_ms: float = 0.0

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.55 and not self.should_rethink

    @property
    def summary(self) -> str:
        flag_labels = [f.value for f in self.flags]
        return (
            f"confidence={self.confidence:.2f}, "
            f"quality={self.reasoning_quality:.2f}, "
            f"flags=[{', '.join(flag_labels)}]"
        )

    def __repr__(self) -> str:
        return f"MetaCognitionReport({self.summary})"


# ─────────────────────────────────────────────────────────────────────────────
# MetaCognition
# ─────────────────────────────────────────────────────────────────────────────

class MetaCognition:
    """
    Anterior PFC analogue: evaluates response quality and confidence.

    Runs AFTER the LLM responds and AFTER ExecutiveMonitor validation.
    Uses heuristic signals to produce a calibrated confidence estimate
    and actionable quality flags.

    Confidence signals (all [0, 1]):
        + Memory support   — more retrieved memories → higher confidence
        + Response length  — too short is suspicious; optimal range is rewarded
        + Structural cues  — numbered lists, specifics, cited reasoning
        - Hedge density    — frequent "maybe/perhaps/I think" → lower confidence
        - Conflict penalty — monitor-detected conflicts → lower confidence
        - Contradiction    — self-contradiction patterns → lower confidence

    Usage:
        meta = MetaCognition()
        report = meta.evaluate(
            response=llm_response,
            validation=monitor_result,
            memories_retrieved=3,
            route="system_2",
        )
        if report.should_rethink:
            # trigger additional correction pass
    """

    # Hedge phrases — these reduce confidence
    _HEDGE_PATTERNS: List[str] = [
        r"\bi think\b", r"\bi believe\b", r"\bi suppose\b",
        r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b", r"\bprobably\b",
        r"\bit seems\b", r"\bit appears\b", r"\bseems like\b",
        r"\bnot sure\b", r"\buncertain\b", r"\bnot certain\b",
        r"\bi\'m not\b", r"\bi am not sure\b",
        r"\bcould be\b", r"\bmight be\b", r"\bmay be\b",
        r"\bas far as i know\b", r"\bto my knowledge\b",
        r"\bi don\'t know\b", r"\bi cannot\b", r"\bi can\'t\b",
    ]

    # Structural confidence boosters
    _STRUCTURE_PATTERNS: List[Tuple[str, float]] = [
        (r"\d+\.",                   0.10),  # Numbered list
        (r"\bbecause\b",             0.08),  # Causal reasoning
        (r"\btherefore\b",           0.10),
        (r"\bfor example\b",         0.08),
        (r"\bspecifically\b",        0.08),
        (r"\bin summary\b",          0.05),
        (r"\bto conclude\b",         0.05),
        (r"\bthe reason\b",          0.07),
        (r"\baccording to\b",        0.10),
        (r"\bthe answer is\b",       0.12),
        (r"\bthe solution is\b",     0.12),
    ]

    # Self-contradiction patterns
    _CONTRADICTION_PATTERNS: List[Tuple[str, str]] = [
        (r"\bis\b",     r"\bis not\b"),
        (r"\bcan\b",    r"\bcannot\b"),
        (r"\bwill\b",   r"\bwill not\b"),
        (r"\byes\b",    r"\bno\b"),
        (r"\btrue\b",   r"\bfalse\b"),
    ]

    def __init__(
        self,
        rethink_threshold: float = 0.30,
        optimal_response_words: Tuple[int, int] = (40, 400),
        verbose: bool = False,
    ):
        """
        Args:
            rethink_threshold: Confidence below this triggers should_rethink=True.
            optimal_response_words: (min, max) word count for ideal responses.
            verbose: Log meta reports.
        """
        self.rethink_threshold       = rethink_threshold
        self.optimal_min, self.optimal_max = optimal_response_words
        self.verbose                 = verbose
        self._eval_count             = 0
        self._rethink_count          = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        response: str,
        validation_passed: bool = True,
        conflict_count: int = 0,
        memories_retrieved: int = 0,
        route: str = "system_1",
    ) -> MetaCognitionReport:
        """
        Evaluate response quality and calibrate confidence.

        Args:
            response: The LLM's response text.
            validation_passed: Whether ExecutiveMonitor passed the response.
            conflict_count: Number of conflicts detected by the monitor.
            memories_retrieved: How many long-term memories were injected.
            route: "system_1" or "system_2" — affects baseline confidence.

        Returns:
            MetaCognitionReport with confidence, quality, and flags.
        """
        t0 = time.perf_counter()
        self._eval_count += 1

        words    = response.split()
        word_cnt = len(words)

        # ── Component scores ──
        hedge_density  = self._score_hedges(response)
        struct_bonus   = self._score_structure(response)
        memory_score   = self._score_memory_support(memories_retrieved, route)
        length_score   = self._score_length(word_cnt)
        conflict_score = self._score_conflicts(conflict_count, validation_passed)
        contra_penalty = self._score_contradiction(response)

        # ── Base confidence from route ──
        base = 0.65 if route == "system_2" else 0.50

        # ── Composite confidence ──
        confidence = (
            base
            + memory_score   * 0.20
            + length_score   * 0.10
            + struct_bonus   * 0.15
            - hedge_density  * 0.25
            + conflict_score * 0.20
            - contra_penalty * 0.15
        )
        confidence = max(0.0, min(1.0, confidence))

        # ── Reasoning quality (structure-focused) ──
        reasoning_quality = (
            length_score   * 0.35
            + struct_bonus * 0.40
            + (1.0 - hedge_density) * 0.25
        )
        reasoning_quality = max(0.0, min(1.0, reasoning_quality))

        # ── Quality flags ──
        flags = self._compute_flags(
            hedge_density=hedge_density,
            word_cnt=word_cnt,
            memories_retrieved=memories_retrieved,
            conflict_count=conflict_count,
            validation_passed=validation_passed,
            confidence=confidence,
            contra_penalty=contra_penalty,
        )

        should_rethink = confidence < self.rethink_threshold
        if should_rethink:
            self._rethink_count += 1

        annotation = self._annotation(confidence)

        report = MetaCognitionReport(
            confidence=round(confidence, 4),
            reasoning_quality=round(reasoning_quality, 4),
            flags=flags,
            hedge_density=round(hedge_density, 4),
            response_length=word_cnt,
            memory_hits=memories_retrieved,
            conflict_count=conflict_count,
            should_rethink=should_rethink,
            annotation=annotation,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

        if self.verbose:
            logger.debug(f"[MetaCognition] {report}")

        return report

    # ─────────────────────────────────────────────
    # Component Scorers
    # ─────────────────────────────────────────────

    def _score_hedges(self, text: str) -> float:
        """Density of hedging phrases [0, 1]. Higher = more uncertain language."""
        words = text.split()
        if not words:
            return 0.5
        matches = sum(
            1 for p in self._HEDGE_PATTERNS
            if re.search(p, text, re.I)
        )
        # Normalize: 5+ distinct hedge patterns is high uncertainty
        raw = matches / max(len(self._HEDGE_PATTERNS), 1)
        return min(raw * 3.0, 1.0)  # Amplify

    def _score_structure(self, text: str) -> float:
        """Structural reasoning cues [0, 1]."""
        score = 0.0
        for pattern, weight in self._STRUCTURE_PATTERNS:
            if re.search(pattern, text, re.I):
                score += weight
        return min(score, 1.0)

    def _score_memory_support(self, memories: int, route: str) -> float:
        """Memory retrieval support [0, 1]."""
        if route == "system_1":
            return 0.4  # Fast path doesn't rely on deep memory
        # System 2: more memories = more grounded
        return min(memories / 5.0, 1.0)

    def _score_length(self, word_cnt: int) -> float:
        """Reward responses in the optimal word-count range."""
        if word_cnt < self.optimal_min:
            # Too short — penalize proportionally
            return word_cnt / self.optimal_min * 0.5
        if word_cnt > self.optimal_max:
            # Too long — mild penalty for verbosity
            excess = word_cnt - self.optimal_max
            return max(1.0 - excess / (self.optimal_max * 2), 0.4)
        # In range — linear reward
        return 0.5 + 0.5 * (
            (word_cnt - self.optimal_min) / (self.optimal_max - self.optimal_min)
        )

    def _score_conflicts(self, conflict_count: int, passed: bool) -> float:
        """Conflict penalty [-1, 0]. More conflicts = lower confidence."""
        if passed and conflict_count == 0:
            return 0.05  # Slight bonus for clean validation
        return -min(conflict_count * 0.15, 0.50)

    def _score_contradiction(self, text: str) -> float:
        """Detect self-contradiction in the response [0, 1]."""
        sentences = re.split(r"[.!?]", text.lower())
        penalty = 0.0
        for pos_pat, neg_pat in self._CONTRADICTION_PATTERNS:
            pos_sents = [s for s in sentences if re.search(pos_pat, s)]
            neg_sents = [s for s in sentences if re.search(neg_pat, s)]
            if pos_sents and neg_sents:
                penalty += 0.25
        return min(penalty, 1.0)

    # ─────────────────────────────────────────────
    # Flags
    # ─────────────────────────────────────────────

    def _compute_flags(
        self,
        hedge_density: float,
        word_cnt: int,
        memories_retrieved: int,
        conflict_count: int,
        validation_passed: bool,
        confidence: float,
        contra_penalty: float,
    ) -> List[QualityFlag]:
        flags = []

        if hedge_density > 0.35:
            flags.append(QualityFlag.OVER_HEDGED)
        elif hedge_density > 0.15:
            flags.append(QualityFlag.UNCERTAIN)

        if word_cnt < self.optimal_min:
            flags.append(QualityFlag.SHALLOW)

        if word_cnt > self.optimal_max * 1.5:
            flags.append(QualityFlag.VERBOSE)

        if contra_penalty > 0.3:
            flags.append(QualityFlag.CONTRADICTORY)

        if memories_retrieved == 0:
            flags.append(QualityFlag.UNGROUNDED)
        elif memories_retrieved <= 1:
            flags.append(QualityFlag.LOW_MEMORY_USE)

        if not validation_passed or conflict_count > 0:
            flags.append(QualityFlag.CONFLICT_FAILED)

        if confidence >= 0.72 and not flags:
            flags.append(QualityFlag.HIGH_CONFIDENCE)

        return flags

    # ─────────────────────────────────────────────
    # Annotation
    # ─────────────────────────────────────────────

    def _annotation(self, confidence: float) -> str:
        if confidence >= 0.80:
            return "High confidence"
        if confidence >= 0.60:
            return "Moderate confidence"
        if confidence >= 0.40:
            return "Low confidence — treat with caution"
        return "Very low confidence — consider rethinking"

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "total_evaluations": self._eval_count,
            "rethink_triggers": self._rethink_count,
            "rethink_rate": round(
                self._rethink_count / self._eval_count, 4
            ) if self._eval_count else 0.0,
        }

    def __repr__(self) -> str:
        return f"MetaCognition(evaluations={self._eval_count})"