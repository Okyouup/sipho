"""
GoalStack: Persistent goal management for Aegis-1 (Phase 5).

v2 improvements:
- Semantic completion detection via optional embed_fn (cosine similarity)
- Goals are pre-embedded at push time — no repeated encoding
- Minimum 2-turn dwell guard prevents instant mis-completion
- Keyword signals still fire first; semantic is the fallback
"""

import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class GoalStatus(Enum):
    ACTIVE    = "active"
    COMPLETED = "completed"
    EXPIRED   = "expired"
    PAUSED    = "paused"


class GoalPriority(Enum):
    LOW      = 1
    NORMAL   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class Goal:
    id: str
    text: str
    priority: GoalPriority
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    ttl_seconds: Optional[float] = None
    completion_signals: List[str] = field(default_factory=list)
    turn_created: int = 0
    turn_completed: Optional[int] = None
    metadata: Dict = field(default_factory=dict)
    _embedding: Optional[List[float]] = field(default=None, repr=False)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.ttl_seconds is not None and self.age_seconds > self.ttl_seconds

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE

    def complete(self, turn: Optional[int] = None) -> None:
        self.status       = GoalStatus.COMPLETED
        self.completed_at = time.time()
        self.turn_completed = turn

    def to_dict(self) -> Dict:
        return {
            "id":             self.id,
            "text":           self.text,
            "priority":       self.priority.name,
            "status":         self.status.value,
            "age_seconds":    round(self.age_seconds, 1),
            "turn_created":   self.turn_created,
            "turn_completed": self.turn_completed,
        }

    def __repr__(self) -> str:
        return f"Goal({self.priority.name}: '{self.text[:50]}' [{self.status.value}])"


# ─────────────────────────────────────────────────────────────────────────────
# GoalStack
# ─────────────────────────────────────────────────────────────────────────────

class GoalStack:
    """
    Dorsolateral PFC analogue: maintains active goals across conversation turns.

    Goals are injected into the LLM system prompt as a "current objectives"
    section. Completed goals are archived.

    v2: optional embed_fn enables semantic completion detection via cosine
    similarity, eliminating false positives from incidental keyword matches
    and false negatives from paraphrased completions.
    """

    def __init__(
        self,
        max_active_goals: int = 10,
        auto_expire: bool = True,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        semantic_threshold: float = 0.62,
        min_dwell_turns: int = 2,
        verbose: bool = False,
    ):
        self.max_active_goals  = max_active_goals
        self.auto_expire       = auto_expire
        self.embed_fn          = embed_fn
        self.semantic_threshold = semantic_threshold
        self.min_dwell_turns   = min_dwell_turns
        self.verbose           = verbose

        self._goals: Dict[str, Goal] = {}
        self._archived: List[Goal]   = []
        self._turn = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def push(
        self,
        text: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        ttl_seconds: Optional[float] = None,
        completion_signals: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Goal:
        if len(self.active_goals) >= self.max_active_goals:
            self._drop_lowest_priority()

        embedding = None
        if self.embed_fn:
            try:
                embedding = self.embed_fn(text)
            except Exception:
                pass

        goal = Goal(
            id=str(uuid.uuid4())[:8],
            text=text,
            priority=priority,
            ttl_seconds=ttl_seconds,
            completion_signals=completion_signals or [],
            turn_created=self._turn,
            metadata=metadata or {},
            _embedding=embedding,
        )
        self._goals[goal.id] = goal

        if self.verbose:
            logger.debug(f"[GoalStack] Pushed: {goal}")
        return goal

    def complete(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        self._goals[goal_id].complete(turn=self._turn)
        self._archive(self._goals[goal_id])
        return True

    def pause(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        self._goals[goal_id].status = GoalStatus.PAUSED
        return True

    def resume(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        if self._goals[goal_id].status == GoalStatus.PAUSED:
            self._goals[goal_id].status = GoalStatus.ACTIVE
        return True

    def remove(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        self._archive(self._goals[goal_id])
        return True

    def check_completion(
        self,
        user_input: str,
        llm_response: str,
        turn: Optional[int] = None,
    ) -> List[Goal]:
        """
        Auto-detect which goals were completed this turn.

        Strategy:
        1. Explicit keyword signals (fast, precise when signals are well-chosen)
        2. Semantic cosine similarity (requires embed_fn; catches paraphrased completions)

        A goal needs ≥ min_dwell_turns turns of activity before semantic
        completion fires, preventing instant mis-completion.
        """
        if turn is not None:
            self._turn = turn

        combined = (user_input + " " + llm_response).lower()
        completed = []

        response_embedding = None
        if self.embed_fn:
            try:
                response_embedding = self.embed_fn(llm_response)
            except Exception:
                pass

        for goal in list(self.active_goals):
            # 1. Keyword signals
            signal_hit = any(sig.lower() in combined for sig in goal.completion_signals)
            if signal_hit:
                if self.verbose:
                    logger.debug(f"[GoalStack] Completed via keyword: {goal}")
                goal.complete(turn=self._turn)
                self._archive(goal)
                completed.append(goal)
                continue

            # 2. Semantic similarity
            if (
                self.embed_fn
                and response_embedding is not None
                and goal._embedding is not None
                and (self._turn - goal.turn_created) >= self.min_dwell_turns
            ):
                sim = self._cosine(goal._embedding, response_embedding)
                if sim >= self.semantic_threshold:
                    if self.verbose:
                        logger.debug(
                            f"[GoalStack] Completed via semantic "
                            f"(sim={sim:.3f}): {goal}"
                        )
                    goal.complete(turn=self._turn)
                    self._archive(goal)
                    completed.append(goal)

        # Expiry sweep
        if self.auto_expire:
            for goal in list(self.active_goals):
                if goal.is_expired:
                    goal.status = GoalStatus.EXPIRED
                    self._archive(goal)
                    if self.verbose:
                        logger.debug(f"[GoalStack] Expired: {goal}")

        return completed

    def tick(self, turn: int) -> None:
        self._turn = turn
        if self.auto_expire:
            for goal in list(self.active_goals):
                if goal.is_expired:
                    goal.status = GoalStatus.EXPIRED
                    self._archive(goal)

    # ─────────────────────────────────────────────
    # Context Injection
    # ─────────────────────────────────────────────

    def get_context_string(self) -> str:
        active = self.active_goals
        if not active:
            return ""
        sorted_goals = sorted(active, key=lambda g: g.priority.value, reverse=True)
        _MARKERS = {
            GoalPriority.CRITICAL: "🔴",
            GoalPriority.HIGH:     "🟠",
            GoalPriority.NORMAL:   "🟡",
            GoalPriority.LOW:      "⚪",
        }
        lines = [
            f"  {_MARKERS.get(g.priority, '•')} [{g.priority.name}] {g.text}"
            for g in sorted_goals
        ]
        return "## Active Goals\nThe following objectives should guide your response:\n" + "\n".join(lines)

    def active_texts(self) -> List[str]:
        return [g.text for g in self.active_goals]

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def active_goals(self) -> List[Goal]:
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    @property
    def all_goals(self) -> List[Goal]:
        return list(self._goals.values())

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _archive(self, goal: Goal) -> None:
        if goal.id in self._goals:
            del self._goals[goal.id]
        self._archived.append(goal)

    def _drop_lowest_priority(self) -> None:
        active = self.active_goals
        if not active:
            return
        weakest = min(active, key=lambda g: (g.priority.value, -g.age_seconds))
        self._archive(weakest)
        if self.verbose:
            logger.debug(f"[GoalStack] Capacity drop: {weakest}")

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @property
    def stats(self) -> Dict:
        return {
            "active_goals":    len(self.active_goals),
            "total_pushed":    len(self._goals) + len(self._archived),
            "total_completed": sum(1 for g in self._archived if g.status == GoalStatus.COMPLETED),
            "semantic_enabled": self.embed_fn is not None,
        }

    def __len__(self) -> int:
        return len(self.active_goals)

    def __repr__(self) -> str:
        return f"GoalStack(active={len(self.active_goals)}, semantic={'on' if self.embed_fn else 'off'})"