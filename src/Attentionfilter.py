import re
import math
import time
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPriority(Enum):
    SUPPRESSED = "suppressed"   # Filtered — below salience floor
    LOW        = "low"          # Proceed minimally (System 1)
    NORMAL     = "normal"       # Standard processing
    HIGH       = "high"         # Elevated — extra context retrieval
    CRITICAL   = "critical"     # Maximum — force System 2 + all modules


@dataclass
class AttentionDecision:
    """Result of the AttentionFilter for a single input."""
    priority: AttentionPriority
    salience: float             # [0, 1] composite salience
    urgency: float              # Lexical urgency score
    repetition_penalty: float   # Penalty for recently seen content
    goal_alignment: float       # Match with active goals
    complexity: float           # Linguistic complexity estimate
    suppressed: bool            # True = skip deep processing
    reason: str
    latency_ms: float = 0.0

    @property
    def passes(self) -> bool:
        """True if input should proceed to the gateway."""
        return not self.suppressed

    @property
    def force_system2(self) -> bool:
        """True if this input demands deliberate processing."""
        return self.priority == AttentionPriority.CRITICAL

    def __repr__(self) -> str:
        return (
            f"AttentionDecision(priority={self.priority.value}, "
            f"salience={self.salience:.3f}, suppressed={self.suppressed})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AttentionFilter
# ─────────────────────────────────────────────────────────────────────────────

class AttentionFilter:
    """
    Pre-gateway selective attention filter.

    Scores input salience and classifies it before the SNN gateway runs.
    Suppressed inputs are handled by a lightweight System 1 fast-path.
    Critical inputs bypass the gateway entirely and force System 2.

    Usage:
        attn = AttentionFilter(goal_fn=lambda: goal_stack.active_texts())
        decision = attn.attend("URGENT: the production database is down!")
        if decision.force_system2:
            # bypass gateway, go straight to full deliberate pipeline
    """

    # (pattern, urgency_weight) — higher = more urgent
    _URGENCY_PATTERNS: List[Tuple[str, float]] = [
        (r"\b(urgent|critical|emergency|immediately|asap)\b",       0.95),
        (r"\b(crash|down|broken|fail|error|exception|traceback)\b", 0.85),
        (r"\b(important|must|need|required|crucial|vital)\b",       0.65),
        (r"\b(problem|issue|bug|wrong|incorrect|mistake)\b",        0.55),
        (r"\b(help|assist|please|could you|can you|would you)\b",   0.30),
        (r"[!]{2,}",                                                 0.50),
        (r"\?{2,}",                                                  0.35),
    ]

    # Low-value / phatic expressions → LOW priority, no suppression
    _PHATIC_PATTERNS: List[str] = [
        r"^(hi|hello|hey|howdy|greetings)[.!]?$",
        r"^(ok|okay|sure|alright|got it|sounds good)[.!]?$",
        r"^(thanks|thank you|thx|ty)[.!]?$",
        r"^(bye|goodbye|see you|later|cya)[.!]?$",
        r"^(yes|no|yeah|nope|yep|nah|yup)[.!]?$",
        r"^(good|great|nice|cool|awesome|perfect)[.!]?$",
    ]

    def __init__(
        self,
        suppression_threshold: float = 0.12,
        critical_threshold: float = 0.80,
        high_threshold: float = 0.55,
        normal_threshold: float = 0.30,
        repetition_window: int = 12,
        repetition_penalty: float = 0.45,
        goal_fn: Optional[Callable[[], List[str]]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            suppression_threshold: Salience below this → SUPPRESSED.
            critical_threshold: Salience above this → CRITICAL (force System 2).
            high_threshold: Salience above this → HIGH priority.
            normal_threshold: Salience above this → NORMAL (below → LOW).
            repetition_window: Number of recent inputs to check for repetition.
            repetition_penalty: Salience reduction for repeated inputs [0, 1].
            goal_fn: Callable returning list of active goal strings.
                     Injected by Aegis from the GoalStack.
            verbose: Log attention decisions.
        """
        self.suppression_threshold = suppression_threshold
        self.critical_threshold    = critical_threshold
        self.high_threshold        = high_threshold
        self.normal_threshold      = normal_threshold
        self.repetition_penalty    = repetition_penalty
        self.goal_fn               = goal_fn
        self.verbose               = verbose

        self._recent_hashes: deque = deque(maxlen=repetition_window)
        self._recent_tokens: deque = deque(maxlen=repetition_window)
        self._attend_count = 0
        self._suppressed_count = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def attend(self, text: str) -> AttentionDecision:
        """
        Assess attentional salience of an input.

        Args:
            text: Raw user input string.

        Returns:
            AttentionDecision with priority, salience, and flags.
        """
        t0 = time.perf_counter()
        self._attend_count += 1
        stripped = text.strip()
        lower    = stripped.lower()

        # ── Fast path: phatic expression ──
        for pat in self._PHATIC_PATTERNS:
            if re.match(pat, lower):
                decision = AttentionDecision(
                    priority=AttentionPriority.LOW,
                    salience=0.18,
                    urgency=0.0,
                    repetition_penalty=0.0,
                    goal_alignment=0.0,
                    complexity=0.05,
                    suppressed=False,
                    reason="Phatic expression — low salience but not suppressed.",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._log(decision)
                self._update_history(stripped)
                return decision

        # ── Component scores ──
        urgency     = self._score_urgency(lower)
        rep_penalty = self._score_repetition(stripped)
        goal_align  = self._score_goal_alignment(lower)
        complexity  = self._score_complexity(stripped)

        # ── Composite salience ──
        # Weights: urgency carries most, then goal alignment, then complexity.
        # Repetition is a multiplicative suppressor.
        raw_salience = (
            urgency    * 0.40
            + goal_align * 0.28
            + complexity * 0.20
            + 0.12          # Baseline — any non-phatic input has value
        )
        salience = raw_salience * (1.0 - rep_penalty * 0.6)
        salience = max(0.0, min(1.0, salience))

        # ── Classify priority ──
        if salience < self.suppression_threshold:
            priority   = AttentionPriority.SUPPRESSED
            suppressed = True
            reason     = f"Salience {salience:.3f} below floor {self.suppression_threshold}."
            self._suppressed_count += 1
        elif salience >= self.critical_threshold:
            priority   = AttentionPriority.CRITICAL
            suppressed = False
            reason     = f"Critical salience {salience:.3f} — forcing System 2."
        elif salience >= self.high_threshold:
            priority   = AttentionPriority.HIGH
            suppressed = False
            reason     = f"High salience {salience:.3f} (urgency={urgency:.2f}, goal={goal_align:.2f})."
        elif salience >= self.normal_threshold:
            priority   = AttentionPriority.NORMAL
            suppressed = False
            reason     = f"Normal salience {salience:.3f}."
        else:
            priority   = AttentionPriority.LOW
            suppressed = False
            reason     = f"Low salience {salience:.3f}."

        decision = AttentionDecision(
            priority=priority,
            salience=salience,
            urgency=urgency,
            repetition_penalty=rep_penalty,
            goal_alignment=goal_align,
            complexity=complexity,
            suppressed=suppressed,
            reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self._log(decision)
        self._update_history(stripped)
        return decision

    def set_goal_fn(self, goal_fn: Callable[[], List[str]]) -> None:
        """Wire in a live goal source (called by Aegis after GoalStack init)."""
        self.goal_fn = goal_fn

    # ─────────────────────────────────────────────
    # Component Scorers
    # ─────────────────────────────────────────────

    def _score_urgency(self, text: str) -> float:
        score = 0.0
        for pattern, weight in self._URGENCY_PATTERNS:
            if re.search(pattern, text, re.I):
                score = max(score, weight)
        # Sentence count / word count as a complexity proxy
        if score == 0.0 and len(text.split()) > 5:
            score = 0.30   # Non-trivial content gets a baseline urgency
        return score

    def _score_repetition(self, text: str) -> float:
        h = self._hash(text)
        # Exact repeat
        if h in self._recent_hashes:
            return self.repetition_penalty
        # Soft repeat: token Jaccard with any of the last 5 inputs
        tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))
        for prev_tokens in list(self._recent_tokens)[-5:]:
            if not prev_tokens:
                continue
            union  = len(tokens | prev_tokens)
            inter  = len(tokens & prev_tokens)
            if union > 0 and (inter / union) > 0.75:
                return self.repetition_penalty * 0.6
        return 0.0

    def _score_goal_alignment(self, text: str) -> float:
        if self.goal_fn is None:
            return 0.28  # Neutral baseline
        goals = self.goal_fn()
        if not goals:
            return 0.28
        text_tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))
        best = 0.0
        for goal in goals:
            goal_tokens = set(re.findall(r"\b\w{3,}\b", goal.lower()))
            if not goal_tokens:
                continue
            overlap = len(text_tokens & goal_tokens) / len(goal_tokens)
            best = max(best, overlap)
        return min(best, 1.0)

    def _score_complexity(self, text: str) -> float:
        """Estimate complexity: word count, vocab diversity, punctuation."""
        words = text.split()
        if not words:
            return 0.0
        word_count_score = min(len(words) / 40.0, 1.0)
        unique_ratio     = len(set(w.lower() for w in words)) / len(words)
        clause_count     = len(re.findall(r"[,;:—]", text))
        clause_score     = min(clause_count / 5.0, 1.0)
        return (word_count_score * 0.5 + unique_ratio * 0.3 + clause_score * 0.2)

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _update_history(self, text: str) -> None:
        self._recent_hashes.append(self._hash(text))
        self._recent_tokens.append(
            set(re.findall(r"\b\w{3,}\b", text.lower()))
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]

    def _log(self, decision: AttentionDecision) -> None:
        if self.verbose:
            logger.debug(f"[Attention] {decision} | {decision.reason}")

    @property
    def stats(self) -> Dict:
        total = self._attend_count
        return {
            "total_inputs": total,
            "suppressed_count": self._suppressed_count,
            "suppression_rate": round(self._suppressed_count / total, 4) if total else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"AttentionFilter("
            f"inputs={self._attend_count}, "
            f"suppressed={self._suppressed_count})"
        )