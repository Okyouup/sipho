"""
MetaCognition: Self-assessment of LLM response quality (Phase 6 of Aegis-1).

v2 improvements:
- Semantic query-response relevance via optional embed_fn
- New IRRELEVANT quality flag when response doesn't address the query
- Structural pattern weight reduced (0.15 → 0.07) — style-gaming resistant
- Hard cap: style alone cannot push confidence above 0.70
- OVER_HEDGED threshold raised — moderate hedging is correct epistemic practice
"""

import re
import time
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class QualityFlag(Enum):
    UNCERTAIN       = "uncertain"
    SHALLOW         = "shallow"
    VERBOSE         = "verbose"
    CONTRADICTORY   = "contradictory"
    OVER_HEDGED     = "over_hedged"
    UNGROUNDED      = "ungrounded"
    CONFLICT_FAILED = "conflict_failed"
    HIGH_CONFIDENCE = "high_confidence"
    LOW_MEMORY_USE  = "low_memory_use"
    IRRELEVANT      = "irrelevant"       # v2: response doesn't address the query


@dataclass
class MetaCognitionReport:
    confidence: float
    reasoning_quality: float
    flags: List[QualityFlag]
    hedge_density: float
    response_length: int
    memory_hits: int
    conflict_count: int
    should_rethink: bool
    annotation: str
    latency_ms: float = 0.0
    relevance_score: float = 1.0    # v2: semantic query-response relevance

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.55 and not self.should_rethink

    @property
    def summary(self) -> str:
        flag_labels = [f.value for f in self.flags]
        return (
            f"confidence={self.confidence:.2f}, "
            f"quality={self.reasoning_quality:.2f}, "
            f"relevance={self.relevance_score:.2f}, "
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

    Confidence signals:
        + Memory support      — retrieved memories → higher confidence
        + Response length     — optimal range rewarded
        + Structural cues     — capped at 0.07 (v2: style-gaming resistant)
        + Semantic relevance  — response addresses the actual query (v2)
        - Hedge density       — excessive hedging → lower confidence
        - Conflict penalty    — monitor-detected conflicts
        - Contradiction       — self-contradiction patterns
        - Irrelevance         — response doesn't match query (v2)
    """

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

    _STRUCTURE_PATTERNS: List[Tuple[str, float]] = [
        (r"\d+\.",            0.06),
        (r"\bbecause\b",      0.05),
        (r"\btherefore\b",    0.06),
        (r"\bfor example\b",  0.05),
        (r"\bspecifically\b", 0.05),
        (r"\bin summary\b",   0.03),
        (r"\bto conclude\b",  0.03),
        (r"\bthe reason\b",   0.04),
        (r"\baccording to\b", 0.06),
        (r"\bthe answer is\b",0.07),
        (r"\bthe solution is\b",0.07),
    ]

    _CONTRADICTION_PATTERNS: List[Tuple[str, str]] = [
        (r"\bis\b",   r"\bis not\b"),
        (r"\bcan\b",  r"\bcannot\b"),
        (r"\bwill\b", r"\bwill not\b"),
        (r"\byes\b",  r"\bno\b"),
        (r"\btrue\b", r"\bfalse\b"),
    ]

    _STRUCTURE_CAP        = 0.07   # max confidence contribution from style alone
    _IRRELEVANCE_THRESHOLD = 0.40  # relevance below this → IRRELEVANT flag

    def __init__(
        self,
        rethink_threshold: float = 0.30,
        optimal_response_words: Tuple[int, int] = (40, 400),
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        verbose: bool = False,
    ):
        self.rethink_threshold         = rethink_threshold
        self.optimal_min, self.optimal_max = optimal_response_words
        self.embed_fn                  = embed_fn
        self.verbose                   = verbose
        self._eval_count               = 0
        self._rethink_count            = 0

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
        user_query: Optional[str] = None,
    ) -> MetaCognitionReport:
        t0 = time.perf_counter()
        self._eval_count += 1

        words    = response.split()
        word_cnt = len(words)

        hedge_density  = self._score_hedges(response)
        struct_bonus   = self._score_structure(response)
        memory_score   = self._score_memory_support(memories_retrieved, route)
        length_score   = self._score_length(word_cnt)
        conflict_score = self._score_conflicts(conflict_count, validation_passed)
        contra_penalty = self._score_contradiction(response)
        relevance      = self._score_relevance(user_query, response)

        base = 0.65 if route == "system_2" else 0.50

        confidence = (
            base
            + memory_score   * 0.20
            + length_score   * 0.10
            + struct_bonus   * 0.07
            + relevance      * 0.12
            - hedge_density  * 0.22
            + conflict_score * 0.20
            - contra_penalty * 0.15
        )

        # Style-gaming cap
        if memories_retrieved == 0 and relevance < 0.5:
            confidence = min(confidence, 0.70)

        confidence = max(0.0, min(1.0, confidence))

        reasoning_quality = max(0.0, min(1.0,
            length_score   * 0.30
            + struct_bonus * 0.20
            + relevance    * 0.25
            + (1.0 - hedge_density) * 0.25
        ))

        flags = self._compute_flags(
            hedge_density=hedge_density,
            word_cnt=word_cnt,
            memories_retrieved=memories_retrieved,
            conflict_count=conflict_count,
            validation_passed=validation_passed,
            confidence=confidence,
            contra_penalty=contra_penalty,
            relevance_score=relevance,
        )

        should_rethink = confidence < self.rethink_threshold
        if should_rethink:
            self._rethink_count += 1

        report = MetaCognitionReport(
            confidence=round(confidence, 4),
            reasoning_quality=round(reasoning_quality, 4),
            flags=flags,
            hedge_density=round(hedge_density, 4),
            response_length=word_cnt,
            memory_hits=memories_retrieved,
            conflict_count=conflict_count,
            should_rethink=should_rethink,
            annotation=self._annotation(confidence),
            latency_ms=(time.perf_counter() - t0) * 1000,
            relevance_score=round(relevance, 4),
        )

        if self.verbose:
            logger.debug(f"[MetaCognition] {report}")

        return report

    # ─────────────────────────────────────────────
    # Component Scorers
    # ─────────────────────────────────────────────

    def _score_hedges(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.5
        matches = sum(1 for p in self._HEDGE_PATTERNS if re.search(p, text, re.I))
        return min(matches / max(len(self._HEDGE_PATTERNS), 1) * 3.0, 1.0)

    def _score_structure(self, text: str) -> float:
        score = sum(w for p, w in self._STRUCTURE_PATTERNS if re.search(p, text, re.I))
        return min(score, self._STRUCTURE_CAP)

    def _score_memory_support(self, memories: int, route: str) -> float:
        if route == "system_1":
            return 0.4
        return min(memories / 5.0, 1.0)

    def _score_length(self, word_cnt: int) -> float:
        if word_cnt < self.optimal_min:
            return word_cnt / self.optimal_min * 0.5
        if word_cnt > self.optimal_max:
            return max(1.0 - (word_cnt - self.optimal_max) / (self.optimal_max * 2), 0.4)
        return 0.5 + 0.5 * (word_cnt - self.optimal_min) / (self.optimal_max - self.optimal_min)

    def _score_conflicts(self, conflict_count: int, passed: bool) -> float:
        if passed and conflict_count == 0:
            return 0.05
        return -min(conflict_count * 0.15, 0.50)

    def _score_contradiction(self, text: str) -> float:
        sentences = re.split(r"[.!?]", text.lower())
        penalty   = 0.0
        for pos_pat, neg_pat in self._CONTRADICTION_PATTERNS:
            if (any(re.search(pos_pat, s) for s in sentences) and
                    any(re.search(neg_pat, s) for s in sentences)):
                penalty += 0.25
        return min(penalty, 1.0)

    def _score_relevance(self, user_query: Optional[str], response: str) -> float:
        """
        Cosine similarity between query and response embeddings.
        Returns 1.0 (neutral) when embed_fn is unavailable or query is absent.
        """
        if not self.embed_fn or not user_query or not response:
            return 1.0
        try:
            q  = self.embed_fn(user_query)
            r  = self.embed_fn(response)
            dot   = sum(a * b for a, b in zip(q, r))
            mag_q = sum(x * x for x in q) ** 0.5
            mag_r = sum(x * x for x in r) ** 0.5
            if mag_q == 0 or mag_r == 0:
                return 1.0
            return max(0.0, min(1.0, dot / (mag_q * mag_r)))
        except Exception:
            return 1.0

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
        relevance_score: float = 1.0,
    ) -> List[QualityFlag]:
        flags = []

        if hedge_density > 0.45:
            flags.append(QualityFlag.OVER_HEDGED)
        elif hedge_density > 0.20:
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

        if relevance_score < self._IRRELEVANCE_THRESHOLD:
            flags.append(QualityFlag.IRRELEVANT)

        if confidence >= 0.72 and QualityFlag.IRRELEVANT not in flags and not flags:
            flags.append(QualityFlag.HIGH_CONFIDENCE)

        return flags

    def _annotation(self, confidence: float) -> str:
        if confidence >= 0.80: return "High confidence"
        if confidence >= 0.60: return "Moderate confidence"
        if confidence >= 0.40: return "Low confidence — treat with caution"
        return "Very low confidence — consider rethinking"

    # ─────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        return {
            "total_evaluations": self._eval_count,
            "rethink_triggers":  self._rethink_count,
            "rethink_rate":      round(self._rethink_count / self._eval_count, 4) if self._eval_count else 0.0,
            "semantic_enabled":  self.embed_fn is not None,
        }

    def __repr__(self) -> str:
        return f"MetaCognition(evaluations={self._eval_count})"