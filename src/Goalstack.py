import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

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
    """A single goal in the active goal stack."""
    id: str
    text: str                           # Natural language goal description
    priority: GoalPriority
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    ttl_seconds: Optional[float] = None   # None = no expiry
    completion_signals: List[str] = field(default_factory=list)
        # Keywords/phrases that indicate this goal was achieved
    turn_created: int = 0
    turn_completed: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE

    def complete(self, turn: Optional[int] = None) -> None:
        self.status = GoalStatus.COMPLETED
        self.completed_at = time.time()
        self.turn_completed = turn

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "priority": self.priority.name,
            "status": self.status.value,
            "age_seconds": round(self.age_seconds, 1),
            "turn_created": self.turn_created,
            "turn_completed": self.turn_completed,
        }

    def __repr__(self) -> str:
        return (
            f"Goal({self.priority.name}: '{self.text[:50]}' "
            f"[{self.status.value}])"
        )


# ─────────────────────────────────────────────────────────────────────────────
# GoalStack
# ─────────────────────────────────────────────────────────────────────────────

class GoalStack:
    """
    Dorsolateral PFC analogue: maintains active goals across conversation turns.

    Goals are injected into the LLM system prompt as a "current objectives"
    section, biasing responses toward goal completion. Completed goals are
    archived and contribute to long-term episodic memory.

    Usage:
        goals = GoalStack()
        goals.push("Help the user debug their Python script.", priority=GoalPriority.HIGH)
        goals.push("Maintain a friendly and patient tone.", priority=GoalPriority.NORMAL)

        # In Aegis.think():
        extra_context = goals.get_context_string()
        goals.check_completion(user_input, llm_response, turn=5)
    """

    def __init__(
        self,
        max_active_goals: int = 10,
        auto_expire: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            max_active_goals: Maximum concurrent active goals.
            auto_expire: Automatically expire goals past their TTL.
            verbose: Log goal state changes.
        """
        self.max_active_goals = max_active_goals
        self.auto_expire      = auto_expire
        self.verbose          = verbose

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
        """
        Add a new goal to the stack.

        Args:
            text: Natural language description of the goal.
            priority: Goal priority (affects ordering in context injection).
            ttl_seconds: Optional time-to-live in seconds.
            completion_signals: Keywords in LLM responses that indicate completion.
            metadata: Optional key-value context.

        Returns:
            The created Goal object.
        """
        # Prune if at capacity
        if len(self.active_goals) >= self.max_active_goals:
            self._drop_lowest_priority()

        goal = Goal(
            id=str(uuid.uuid4())[:8],
            text=text,
            priority=priority,
            ttl_seconds=ttl_seconds,
            completion_signals=completion_signals or [],
            turn_created=self._turn,
            metadata=metadata or {},
        )
        self._goals[goal.id] = goal

        if self.verbose:
            logger.debug(f"[GoalStack] Pushed: {goal}")

        return goal

    def complete(self, goal_id: str) -> bool:
        """Manually mark a goal as completed."""
        if goal_id not in self._goals:
            return False
        goal = self._goals[goal_id]
        goal.complete(turn=self._turn)
        self._archive(goal)
        if self.verbose:
            logger.debug(f"[GoalStack] Completed: {goal}")
        return True

    def pause(self, goal_id: str) -> bool:
        """Pause a goal (keep it but don't inject it)."""
        if goal_id not in self._goals:
            return False
        self._goals[goal_id].status = GoalStatus.PAUSED
        return True

    def resume(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        if goal_id not in self._goals:
            return False
        if self._goals[goal_id].status == GoalStatus.PAUSED:
            self._goals[goal_id].status = GoalStatus.ACTIVE
        return True

    def remove(self, goal_id: str) -> bool:
        """Remove a goal entirely."""
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
        Auto-detect which goals may have been completed based on the
        LLM response content.

        Checks:
        1. Explicit completion signals (keyword match)
        2. Heuristic: response directly addresses the goal topic

        Args:
            user_input: The user's message.
            llm_response: The LLM's response.
            turn: Current conversation turn index.

        Returns:
            List of goals that were auto-completed.
        """
        if turn is not None:
            self._turn = turn

        combined = (user_input + " " + llm_response).lower()
        completed = []

        for goal in list(self.active_goals):
            # 1. Explicit completion signals
            for signal in goal.completion_signals:
                if signal.lower() in combined:
                    goal.complete(turn=self._turn)
                    self._archive(goal)
                    completed.append(goal)
                    if self.verbose:
                        logger.debug(
                            f"[GoalStack] Auto-completed (signal '{signal}'): {goal}"
                        )
                    break

        # 2. Expiry sweep
        if self.auto_expire:
            for goal in list(self.active_goals):
                if goal.is_expired:
                    goal.status = GoalStatus.EXPIRED
                    self._archive(goal)
                    if self.verbose:
                        logger.debug(f"[GoalStack] Expired: {goal}")

        return completed

    def tick(self, turn: int) -> None:
        """Advance the turn counter (called by Aegis each cycle)."""
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
        """
        Build a context block to inject into the LLM system prompt.
        Returns an empty string if there are no active goals.
        """
        active = self.active_goals
        if not active:
            return ""

        # Sort by priority descending
        sorted_goals = sorted(active, key=lambda g: g.priority.value, reverse=True)

        lines = []
        for g in sorted_goals:
            marker = {
                GoalPriority.CRITICAL: "🔴",
                GoalPriority.HIGH:     "🟠",
                GoalPriority.NORMAL:   "🟡",
                GoalPriority.LOW:      "⚪",
            }.get(g.priority, "•")
            lines.append(f"  {marker} [{g.priority.name}] {g.text}")

        return (
            "## Active Goals\n"
            "The following objectives should guide your response:\n"
            + "\n".join(lines)
        )

    def active_texts(self) -> List[str]:
        """Return text of all active goals (for AttentionFilter.goal_fn)."""
        return [g.text for g in self.active_goals]

    # ─────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────

    @property
    def active_goals(self) -> List[Goal]:
        return [
            g for g in self._goals.values()
            if g.status == GoalStatus.ACTIVE
        ]

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
        """Remove the lowest-priority active goal to make room."""
        active = self.active_goals
        if not active:
            return
        weakest = min(active, key=lambda g: (g.priority.value, -g.age_seconds))
        self._archive(weakest)
        if self.verbose:
            logger.debug(f"[GoalStack] Capacity drop: {weakest}")

    @property
    def stats(self) -> Dict:
        return {
            "active_goals": len(self.active_goals),
            "total_pushed": len(self._goals) + len(self._archived),
            "total_completed": sum(1 for g in self._archived if g.status == GoalStatus.COMPLETED),
            "total_expired": sum(1 for g in self._archived if g.status == GoalStatus.EXPIRED),
            "current_turn": self._turn,
        }

    def __len__(self) -> int:
        return len(self.active_goals)

    def __repr__(self) -> str:
        return f"GoalStack(active={len(self.active_goals)}, archived={len(self._archived)})"