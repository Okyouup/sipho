"""
AttentionFilter: Pre-gateway selective attention filter (Phase 0 of Aegis-1).

v2 improvements:
- Negation-aware urgency: "not an emergency" no longer triggers CRITICAL
- Fictional / hypothetical framing detection with a 5-message context window
- Sentence-boundary scoping prevents cross-sentence negation bleed
"""

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
    SUPPRESSED = "suppressed"
    LOW        = "low"
    NORMAL     = "normal"
    HIGH       = "high"
    CRITICAL   = "critical"


@dataclass
class AttentionDecision:
    priority: AttentionPriority
    salience: float
    urgency: float
    repetition_penalty: float
    goal_alignment: float
    complexity: float
    suppressed: bool
    reason: str
    latency_ms: float = 0.0
    fictional_context: bool = False
    urgency_negated: bool = False

    @property
    def passes(self) -> bool:
        return not self.suppressed

    @property
    def force_system2(self) -> bool:
        return self.priority == AttentionPriority.CRITICAL

    def __repr__(self) -> str:
        flags = []
        if self.fictional_context: flags.append("fiction")
        if self.urgency_negated:   flags.append("negated")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        return (
            f"AttentionDecision(priority={self.priority.value}, "
            f"salience={self.salience:.3f}, suppressed={self.suppressed}{flag_str})"
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

    v2: negation-aware urgency scoring, fictional framing detection with
    a rolling context window, and sentence-boundary negation scoping.
    """

    _URGENCY_PATTERNS: List[Tuple[str, float]] = [
        (r"\b(urgent|critical|emergency|immediately|asap)\b",        0.95),
        (r"\b(crash|down|broken|fail|error|exception|traceback)\b",  0.85),
        (r"\b(important|must|need|required|crucial|vital)\b",        0.65),
        (r"\b(problem|issue|bug|wrong|incorrect|mistake)\b",         0.55),
        (r"\b(help|assist|please|could you|can you|would you)\b",    0.30),
        (r"[!]{2,}",                                                  0.50),
        (r"\?{2,}",                                                   0.35),
    ]

    _PHATIC_PATTERNS: List[str] = [
        r"^(hi|hello|hey|howdy|greetings)[.!]?$",
        r"^(ok|okay|sure|alright|got it|sounds good)[.!]?$",
        r"^(thanks|thank you|thx|ty)[.!]?$",
        r"^(bye|goodbye|see you|later|cya)[.!]?$",
        r"^(yes|no|yeah|nope|yep|nah|yup)[.!]?$",
        r"^(good|great|nice|cool|awesome|perfect)[.!]?$",
    ]

    _NEGATION_RE = re.compile(
        r"\b(not|no|never|isn't|aren't|doesn't|don't|won't|can't|couldn't|"
        r"wouldn't|shouldn't|hardly|barely|nothing|without|nor|neither)\b",
        re.I,
    )
    _NEGATION_WINDOW = 50

    _FICTION_RE = re.compile(
        r"\b(writing|wrote|write)\s+(a\s+|an\s+|the\s+)?"
        r"(story|novel|scene|script|character|fiction|narrative|play|screenplay)\b"
        r"|\b(in\s+my|in\s+the)\s+(story|novel|book|game|script|fiction|narrative)\b"
        r"|\bhypothetically\b"
        r"|\bimagine\s+if\b"
        r"|\bfor\s+(a\s+|my\s+|the\s+)?(story|novel|character|game|screenplay)\b"
        r"|\bmy\s+(fictional\s+)?character\b"
        r"|\b(fictional|fictionally|as\s+a\s+joke|roleplay|role[\s-]play)\b"
        r"|\blet's\s+(say|pretend|imagine)\b"
        r"|\bwhat\s+if\s+I\b"
        r"|\bfor\s+a\s+book\b",
        re.I,
    )
    _FICTION_URGENCY_MULT = 0.25

    def __init__(
        self,
        suppression_threshold: float = 0.12,
        critical_threshold: float = 0.80,
        high_threshold: float = 0.55,
        normal_threshold: float = 0.30,
        repetition_window: int = 12,
        repetition_penalty: float = 0.45,
        goal_fn: Optional[Callable[[], List[str]]] = None,
        context_window: int = 5,
        verbose: bool = False,
    ):
        self.suppression_threshold = suppression_threshold
        self.critical_threshold    = critical_threshold
        self.high_threshold        = high_threshold
        self.normal_threshold      = normal_threshold
        self.repetition_penalty    = repetition_penalty
        self.goal_fn               = goal_fn
        self.verbose               = verbose

        self._recent_hashes: deque  = deque(maxlen=repetition_window)
        self._recent_tokens: deque  = deque(maxlen=repetition_window)
        self._context_history: deque = deque(maxlen=context_window)
        self._attend_count    = 0
        self._suppressed_count = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def attend(self, text: str) -> AttentionDecision:
        t0      = time.perf_counter()
        self._attend_count += 1
        stripped = text.strip()
        lower    = stripped.lower()

        # Fast path: phatic
        for pat in self._PHATIC_PATTERNS:
            if re.match(pat, lower):
                decision = AttentionDecision(
                    priority=AttentionPriority.LOW,
                    salience=0.18, urgency=0.0,
                    repetition_penalty=0.0, goal_alignment=0.0,
                    complexity=0.05, suppressed=False,
                    reason="Phatic expression — low salience but not suppressed.",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                self._log(decision)
                self._update_history(stripped)
                return decision

        fictional                 = self._detect_fiction(lower)
        urgency, urgency_negated  = self._score_urgency(lower, fictional)
        rep_penalty               = self._score_repetition(stripped)
        goal_align                = self._score_goal_alignment(lower)
        complexity                = self._score_complexity(stripped)

        raw_salience = (
            urgency    * 0.40
            + goal_align * 0.28
            + complexity * 0.20
            + 0.12
        )
        salience = max(0.0, min(1.0, raw_salience * (1.0 - rep_penalty * 0.6)))

        if salience < self.suppression_threshold:
            priority, suppressed = AttentionPriority.SUPPRESSED, True
            reason = f"Salience {salience:.3f} below floor {self.suppression_threshold}."
            self._suppressed_count += 1
        elif salience >= self.critical_threshold:
            priority, suppressed = AttentionPriority.CRITICAL, False
            reason = f"Critical salience {salience:.3f} — forcing System 2."
        elif salience >= self.high_threshold:
            priority, suppressed = AttentionPriority.HIGH, False
            reason = f"High salience {salience:.3f} (urgency={urgency:.2f}, goal={goal_align:.2f})."
        elif salience >= self.normal_threshold:
            priority, suppressed = AttentionPriority.NORMAL, False
            reason = f"Normal salience {salience:.3f}."
        else:
            priority, suppressed = AttentionPriority.LOW, False
            reason = f"Low salience {salience:.3f}."

        if fictional:       reason += " [fictional framing — urgency dampened]"
        if urgency_negated: reason += " [urgency keyword negated]"

        decision = AttentionDecision(
            priority=priority, salience=salience,
            urgency=urgency, repetition_penalty=rep_penalty,
            goal_alignment=goal_align, complexity=complexity,
            suppressed=suppressed, reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            fictional_context=fictional,
            urgency_negated=urgency_negated,
        )
        self._log(decision)
        self._update_history(stripped)
        return decision

    def set_goal_fn(self, goal_fn: Callable[[], List[str]]) -> None:
        self.goal_fn = goal_fn

    # ─────────────────────────────────────────────
    # Scorers
    # ─────────────────────────────────────────────

    def _detect_fiction(self, text: str) -> bool:
        if self._FICTION_RE.search(text):
            return True
        for prev in self._context_history:
            if self._FICTION_RE.search(prev):
                return True
        return False

    def _score_urgency(self, text: str, fictional: bool) -> Tuple[float, bool]:
        fiction_mult = self._FICTION_URGENCY_MULT if fictional else 1.0
        score        = 0.0
        any_negated  = False

        for pattern, weight in self._URGENCY_PATTERNS:
            for match in re.finditer(pattern, text, re.I):
                neg_mult = self._negation_mult(text, match.start())
                if neg_mult < 1.0:
                    any_negated = True
                score = max(score, weight * neg_mult * fiction_mult)

        if score == 0.0 and len(text.split()) > 5:
            score = 0.30

        return score, any_negated

    def _negation_mult(self, text: str, mstart: int) -> float:
        prefix = text[max(0, mstart - self._NEGATION_WINDOW): mstart]
        last_b = max(prefix.rfind(". "), prefix.rfind("! "),
                     prefix.rfind("? "), prefix.rfind("\n"))
        if last_b != -1:
            prefix = prefix[last_b + 2:]
        return 0.08 if self._NEGATION_RE.search(prefix) else 1.0

    def _score_repetition(self, text: str) -> float:
        h = self._hash(text)
        if h in self._recent_hashes:
            return self.repetition_penalty
        tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))
        for prev in list(self._recent_tokens)[-5:]:
            if not prev:
                continue
            union = len(tokens | prev)
            inter = len(tokens & prev)
            if union > 0 and inter / union > 0.75:
                return self.repetition_penalty * 0.6
        return 0.0

    def _score_goal_alignment(self, text: str) -> float:
        if self.goal_fn is None:
            return 0.28
        goals = self.goal_fn()
        if not goals:
            return 0.28
        text_tokens = set(re.findall(r"\b\w{3,}\b", text.lower()))
        best = 0.0
        for goal in goals:
            goal_tokens = set(re.findall(r"\b\w{3,}\b", goal.lower()))
            if not goal_tokens:
                continue
            best = max(best, len(text_tokens & goal_tokens) / len(goal_tokens))
        return min(best, 1.0)

    def _score_complexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        word_count_score = min(len(words) / 40.0, 1.0)
        unique_ratio     = len(set(w.lower() for w in words)) / len(words)
        clause_count     = len(re.findall(r"[,;:—]", text))
        clause_score     = min(clause_count / 5.0, 1.0)
        return word_count_score * 0.5 + unique_ratio * 0.3 + clause_score * 0.2

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _update_history(self, text: str) -> None:
        self._recent_hashes.append(self._hash(text))
        self._recent_tokens.append(set(re.findall(r"\b\w{3,}\b", text.lower())))
        self._context_history.append(text.lower())

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
            "total_inputs":     total,
            "suppressed_count": self._suppressed_count,
            "suppression_rate": round(self._suppressed_count / total, 4) if total else 0.0,
        }

    def __repr__(self) -> str:
        return f"AttentionFilter(inputs={self._attend_count}, suppressed={self._suppressed_count})"