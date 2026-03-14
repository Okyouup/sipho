import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from Perceptualgateway import PerceptualGateway, Route, GatewayDecision
from Vsamemory         import VSAMemory, QueryResult
from Executivemonitor  import ExecutiveMonitor, ValidationResult, ConflictSeverity
from Cortex            import Cortex
from Attentionfilter   import AttentionFilter, AttentionDecision, AttentionPriority
from Valencetagger     import EmotionalValenceTagger, ValenceTag
from Goalstack         import GoalStack, Goal, GoalPriority
from Metacognition     import MetaCognition, MetaCognitionReport, QualityFlag

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Response Envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AegisResponse:
    """Full response envelope from Aegis-1 — text + complete cognitive telemetry."""
    text: str                           # Final response text
    route: Route                        # Processing path taken
    turn: int                           # Conversation turn index

    # Cognitive telemetry — one field per subsystem
    attention: AttentionDecision        # Phase 0
    gateway: GatewayDecision            # Phase 1
    valence: ValenceTag                 # Phase 4
    validation: ValidationResult        # Phase 3
    meta: MetaCognitionReport           # Phase 6
    goals_completed: List[Goal]         # Goals auto-completed this turn

    rethink_attempts: int = 0
    vsa_hits: int = 0
    cortex_memories: int = 0

    # Timing
    total_latency_ms: float = 0.0
    llm_latency_ms: float   = 0.0
    monitor_latency_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"AegisResponse("
            f"route={self.route.value}, "
            f"turn={self.turn}, "
            f"confidence={self.meta.confidence:.2f}, "
            f"rethinks={self.rethink_attempts}, "
            f"valid={self.validation.passed}, "
            f"latency={self.total_latency_ms:.1f}ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Aegis-1
# ─────────────────────────────────────────────────────────────────────────────

class Aegis:
    """
    The Aegis-1 Cognitive Control Plane.

    Drop-in cognitive middleware for any LLM.  Wire with any
    llm_fn and embed_fn (from Llm.py adapters or custom), then call
    `aegis.think(user_input)`.

    Six cognitive phases wrap every LLM call:
        0. AttentionFilter   — gate low-salience inputs
        1. PerceptualGateway — route System 1 / System 2
        2. VSAMemory         — HDC associative retrieval
        3. ExecutiveMonitor  — hallucination interception
        4. ValenceTagger     — affective encoding weight
        5. GoalStack         — active goal injection
        6. MetaCognition     — confidence & quality evaluation

    Example:
        from Llm import anthropic_adapter
        from Aegis import Aegis

        llm_fn, embed_fn = anthropic_adapter(api_key="sk-ant-...")
        aegis = Aegis(llm_fn=llm_fn, embed_fn=embed_fn, verbose=True)

        # Pre-load domain facts
        aegis.learn_fact("Water boils at 100°C at sea level.", category="physics")

        # Set persistent goals
        aegis.push_goal("Always cite sources when making factual claims.")
        aegis.push_goal("Help the user debug their FastAPI application.",
                        priority=GoalPriority.HIGH)

        # Converse
        response = aegis.think("Why is my API returning 422?")
        print(response.text)
        print(f"Confidence: {response.meta.confidence:.0%}")
    """

    def __init__(
        self,
        llm_fn: Callable[[List[Dict], str], str],
        embed_fn: Callable[[str], List[float]],

        # ── Cortex ──
        memory_path: Optional[str] = None,
        working_memory_capacity: int = 10,
        recall_top_k: int = 5,
        recall_threshold: float = 0.25,

        # ── AttentionFilter ──
        attention_suppression_threshold: float = 0.12,
        attention_critical_threshold: float = 0.80,
        attention_repetition_window: int = 12,

        # ── PerceptualGateway ──
        spike_threshold: float = 0.55,
        novelty_window: int = 50,

        # ── VSA Memory ──
        vsa_dim: int = 10_000,
        vsa_capacity: int = 5_000,
        vsa_threshold: float = 0.55,

        # ── ExecutiveMonitor ──
        knowledge_path: Optional[str] = None,
        monitor_strong_threshold: float = 0.72,
        monitor_moderate_threshold: float = 0.50,
        max_rethink_attempts: int = 2,

        # ── MetaCognition ──
        meta_rethink_threshold: float = 0.30,
        meta_optimal_response_words: Tuple[int, int] = (40, 400),

        # ── General ──
        system_prompt: str = (
            "You are a helpful, precise assistant with access to a cognitive memory system. "
            "You reason carefully and avoid making claims you are uncertain about."
        ),
        verbose: bool = False,
    ):
        self.llm_fn        = llm_fn
        self.system_prompt = system_prompt
        self.verbose       = verbose
        self.max_rethink_attempts = max_rethink_attempts
        self._turn_count   = 0

        # ── Phase 5: GoalStack (init first — AttentionFilter hooks into it) ──
        self.goals = GoalStack(verbose=verbose)

        # ── Phase 0: AttentionFilter ──
        self.attention = AttentionFilter(
            suppression_threshold=attention_suppression_threshold,
            critical_threshold=attention_critical_threshold,
            repetition_window=attention_repetition_window,
            goal_fn=self.goals.active_texts,   # Live hook into GoalStack
            verbose=verbose,
        )

        # ── Phase 4: EmotionalValenceTagger ──
        self.valence = EmotionalValenceTagger(verbose=verbose)

        # ── Phase 1: PerceptualGateway ──
        self.gateway = PerceptualGateway(
            spike_threshold=spike_threshold,
            novelty_window=novelty_window,
            embed_fn=embed_fn,
            verbose=verbose,
        )

        # ── Phase 2: VSA Memory ──
        self.vsa = VSAMemory(
            dim=vsa_dim,
            capacity=vsa_capacity,
            similarity_threshold=vsa_threshold,
            verbose=verbose,
        )

        # ── Phase 3: ExecutiveMonitor ──
        self.monitor = ExecutiveMonitor(
            knowledge_path=knowledge_path,
            embed_fn=embed_fn,
            strong_threshold=monitor_strong_threshold,
            moderate_threshold=monitor_moderate_threshold,
            max_rethink_attempts=max_rethink_attempts,
            verbose=verbose,
        )

        # ── Cortex (Synaptic Memory Layer) ──
        self.cortex = Cortex(
            llm_fn=llm_fn,
            embed_fn=embed_fn,
            memory_path=memory_path,
            working_memory_capacity=working_memory_capacity,
            recall_top_k=recall_top_k,
            recall_threshold=recall_threshold,
            system_prompt=system_prompt,
            auto_encode=True,
            verbose=False,
        )

        # ── Phase 6: MetaCognition ──
        self.meta = MetaCognition(
            rethink_threshold=meta_rethink_threshold,
            optimal_response_words=meta_optimal_response_words,
            verbose=verbose,
        )

        if verbose:
            logger.setLevel(logging.DEBUG)
            logging.basicConfig(
                format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                level=logging.DEBUG,
            )

    # ─────────────────────────────────────────────
    # Main Cognitive Loop
    # ─────────────────────────────────────────────

    def think(
        self,
        user_input: str,
        force_route: Optional[Route] = None,
    ) -> AegisResponse:
        """
        The full Aegis-1 cognitive loop.

        Flow:
            0. AttentionFilter   → salience gate (suppressed → System 1 fast path)
            1. ValenceTagger     → tag emotional weight of input
            2. GoalStack tick    → advance turn, check expiry, get goal context
            3. PerceptualGateway → novelty → System 1 or System 2 route
            4. LLM call          → System 1 (fast) or System 2 (deliberate + VSA)
            5. ExecutiveMonitor  → validate output, rethink if conflicts found
            6. MetaCognition     → evaluate confidence, flag quality issues
            7. GoalStack.check   → auto-complete goals from this exchange
            8. Encode to VSA     → store exchange in HDC memory

        Args:
            user_input: Raw user message.
            force_route: Override gateway routing (for testing).

        Returns:
            AegisResponse with text + full cognitive telemetry.
        """
        t_start = time.perf_counter()
        self._turn_count += 1

        # ── Phase 0: Attention Filter ──
        attention_decision = self.attention.attend(user_input)
        if self.verbose:
            self._log(f"Turn {self._turn_count} | {attention_decision}")

        # ── Phase 4: Emotional Valence Tagging ──
        valence_tag = self.valence.tag(user_input)
        if self.verbose:
            self._log(f"Valence | {valence_tag}")

        # ── Phase 5: GoalStack tick + context ──
        self.goals.tick(self._turn_count)
        goal_context = self.goals.get_context_string()

        # ── Phase 1: Perceptual Gateway ──
        # Attention overrides: suppressed → System 1, critical → System 2
        gateway_decision = self.gateway.assess(user_input)
        if attention_decision.suppressed:
            route = Route.SYSTEM_1
        elif attention_decision.force_system2:
            route = Route.SYSTEM_2
        else:
            route = force_route or gateway_decision.route

        if self.verbose:
            self._log(f"Route={route.value} | Gateway={gateway_decision}")

        # ── Phase 1–2: LLM Call ──
        t_llm = time.perf_counter()
        if route == Route.SYSTEM_1:
            response_text, vsa_hits, cortex_memories = self._system1(
                user_input, goal_context
            )
        else:
            response_text, vsa_hits, cortex_memories = self._system2(
                user_input, goal_context
            )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0

        # ── Phase 3: Executive Monitor ──
        t_monitor = time.perf_counter()
        validation = self.monitor.validate(response_text, context=user_input)
        monitor_ms = (time.perf_counter() - t_monitor) * 1000.0

        # ── Re-think Loop (System 2 only) ──
        rethink_attempts = 0
        if validation.requires_rethink and route == Route.SYSTEM_2:
            response_text, rethink_attempts = self._rethink_loop(
                user_input, response_text, validation
            )
            validation = self.monitor.validate(response_text, context=user_input)

        # ── Phase 6: MetaCognition ──
        meta_report = self.meta.evaluate(
            response=response_text,
            validation_passed=validation.passed,
            conflict_count=len(validation.conflicts),
            memories_retrieved=cortex_memories,
            route=route.value,
        )

        # MetaCognition-triggered rethink (confidence too low, System 2 only)
        if (
            meta_report.should_rethink
            and route == Route.SYSTEM_2
            and rethink_attempts < self.max_rethink_attempts
        ):
            response_text, extra_attempts = self._meta_rethink(
                user_input, response_text, meta_report
            )
            rethink_attempts += extra_attempts
            # Re-evaluate after meta-rethink
            validation  = self.monitor.validate(response_text, context=user_input)
            meta_report = self.meta.evaluate(
                response=response_text,
                validation_passed=validation.passed,
                conflict_count=len(validation.conflicts),
                memories_retrieved=cortex_memories,
                route=route.value,
            )

        # ── Phase 5: Goal Completion Check ──
        completed_goals = self.goals.check_completion(
            user_input, response_text, turn=self._turn_count
        )

        # ── Encode exchange to VSA ──
        # Boost encoding weight for high-arousal emotional content
        self.vsa.store(
            label=f"turn_{self._turn_count}",
            text=f"Q: {user_input[:200]}\nA: {response_text[:400]}",
            metadata={
                "turn": self._turn_count,
                "route": route.value,
                "valence": valence_tag.valence,
                "arousal": valence_tag.arousal,
                "confidence": meta_report.confidence,
            },
        )

        # ── Encode valence-weighted memory in Cortex ──
        # High-arousal events are encoded with boosted synaptic weight
        if valence_tag.memory_weight != 1.0:
            self.cortex.remember(
                f"Q: {user_input[:200]}\nA: {response_text[:400]}",
                metadata={
                    **valence_tag.to_metadata(),
                    "turn": self._turn_count,
                    "type": "affective_memory",
                },
            )

        total_ms = (time.perf_counter() - t_start) * 1000.0

        result = AegisResponse(
            text=response_text,
            route=route,
            turn=self._turn_count,
            attention=attention_decision,
            gateway=gateway_decision,
            valence=valence_tag,
            validation=validation,
            meta=meta_report,
            goals_completed=completed_goals,
            rethink_attempts=rethink_attempts,
            vsa_hits=vsa_hits,
            cortex_memories=cortex_memories,
            total_latency_ms=total_ms,
            llm_latency_ms=llm_ms,
            monitor_latency_ms=monitor_ms,
        )

        if self.verbose:
            self._log(f"Response ready | {result}")

        return result

    # ─────────────────────────────────────────────
    # System 1: Fast Path
    # ─────────────────────────────────────────────

    def _system1(
        self, user_input: str, goal_context: str
    ) -> Tuple[str, int, int]:
        """Fast path: Cortex with minimal overhead. Goals still injected."""
        response = self.cortex.think(
            user_input,
            extra_context=goal_context if goal_context else None,
            recall_top_k=3,
        )
        return response, 0, 3

    # ─────────────────────────────────────────────
    # System 2: Deliberate Path
    # ─────────────────────────────────────────────

    def _system2(
        self, user_input: str, goal_context: str
    ) -> Tuple[str, int, int]:
        """
        Deliberate path: VSA context + goal context + full Cortex retrieval.
        """
        # VSA associative retrieval
        vsa_results: List[QueryResult] = self.vsa.query(user_input, top_k=5)
        vsa_context = self._format_vsa_context(vsa_results)

        # Merge VSA context and goal context
        extra_parts = []
        if vsa_context:
            extra_parts.append(vsa_context)
        if goal_context:
            extra_parts.append(goal_context)
        extra = "\n\n".join(extra_parts) if extra_parts else None

        response = self.cortex.think(
            user_input,
            extra_context=extra,
            recall_top_k=5,
        )
        return response, len(vsa_results), 5

    # ─────────────────────────────────────────────
    # Re-Think Loops
    # ─────────────────────────────────────────────

    def _rethink_loop(
        self,
        user_input: str,
        original_response: str,
        validation: ValidationResult,
    ) -> Tuple[str, int]:
        """Monitor-triggered rethink: conflict correction loop."""
        current_response = original_response
        attempts = 0

        for attempt in range(1, self.max_rethink_attempts + 1):
            attempts = attempt
            if self.verbose:
                self._log(
                    f"Monitor Re-think #{attempt} | "
                    f"conflicts={len(validation.conflicts)} | "
                    f"severity={validation.severity.value}"
                )

            rethink_prompt = validation.rethink_prompt or (
                "Please revise your previous response to be more accurate."
            )
            messages = [
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": current_response},
                {"role": "user",      "content": rethink_prompt},
            ]

            try:
                corrected = self.llm_fn(messages, self.system_prompt)
            except Exception as e:
                logger.error(f"[Aegis] LLM error during monitor rethink: {e}")
                break

            current_response = corrected
            new_validation   = self.monitor.validate(corrected, context=user_input)

            if not new_validation.requires_rethink:
                if self.verbose:
                    self._log(f"Monitor Re-think #{attempt} succeeded.")
                break

            validation = new_validation

        return current_response, attempts

    def _meta_rethink(
        self,
        user_input: str,
        original_response: str,
        meta_report: MetaCognitionReport,
    ) -> Tuple[str, int]:
        """MetaCognition-triggered rethink: confidence improvement loop."""
        if self.verbose:
            self._log(
                f"Meta Re-think triggered | "
                f"confidence={meta_report.confidence:.2f} | "
                f"flags={[f.value for f in meta_report.flags]}"
            )

        flag_advice = self._meta_advice(meta_report)
        rethink_prompt = (
            f"Your previous response may not be fully reliable "
            f"({meta_report.annotation}). {flag_advice} "
            f"Please provide a more thorough and grounded response."
        )
        messages = [
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": original_response},
            {"role": "user",      "content": rethink_prompt},
        ]

        try:
            corrected = self.llm_fn(messages, self.system_prompt)
            return corrected, 1
        except Exception as e:
            logger.error(f"[Aegis] LLM error during meta rethink: {e}")
            return original_response, 0

    @staticmethod
    def _meta_advice(report: MetaCognitionReport) -> str:
        """Build targeted advice from quality flags."""
        advice = []
        if QualityFlag.SHALLOW in report.flags:
            advice.append("The response is too brief — please elaborate.")
        if QualityFlag.OVER_HEDGED in report.flags:
            advice.append("Reduce uncertainty qualifiers and be more direct.")
        if QualityFlag.CONTRADICTORY in report.flags:
            advice.append("Resolve any contradictions in your reasoning.")
        if QualityFlag.UNGROUNDED in report.flags:
            advice.append("Ground your claims in specific facts or examples.")
        return " ".join(advice) if advice else "Please improve the quality and accuracy."

    # ─────────────────────────────────────────────
    # Public Goal API
    # ─────────────────────────────────────────────

    def push_goal(
        self,
        text: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        ttl_seconds: Optional[float] = None,
        completion_signals: Optional[List[str]] = None,
    ) -> Goal:
        """
        Add an active goal that will be injected into every LLM prompt.

        Args:
            text: Natural language goal description.
            priority: GoalPriority.LOW / NORMAL / HIGH / CRITICAL
            ttl_seconds: Auto-expire after this many seconds.
            completion_signals: Keywords that auto-complete this goal.

        Returns:
            Goal object (save the .id to manage it later).
        """
        return self.goals.push(
            text=text,
            priority=priority,
            ttl_seconds=ttl_seconds,
            completion_signals=completion_signals or [],
        )

    def complete_goal(self, goal_id: str) -> bool:
        """Manually mark a goal as completed."""
        return self.goals.complete(goal_id)

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal from the stack."""
        return self.goals.remove(goal_id)

    def list_goals(self) -> List[Goal]:
        """Return all currently active goals."""
        return self.goals.active_goals

    # ─────────────────────────────────────────────
    # Knowledge & Memory Management
    # ─────────────────────────────────────────────

    def learn_fact(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
        also_store_in_vsa: bool = True,
        also_store_in_cortex: bool = True,
    ) -> None:
        """
        Teach Aegis a ground-truth fact.
        Stores it in the Monitor's SKG, VSA memory, and Cortex synaptic memory.
        """
        self.monitor.learn(fact, category=category, confidence=confidence)
        if also_store_in_vsa:
            self.vsa.store(
                fact, fact,
                metadata={"category": category, "type": "fact"},
                topic=category,
            )
        if also_store_in_cortex:
            self.cortex.remember(
                fact, metadata={"category": category, "type": "fact"}
            )

    def learn_facts_batch(self, facts: List[Dict]) -> int:
        for f in facts:
            self.learn_fact(
                fact=f["text"],
                category=f.get("category", "general"),
                confidence=f.get("confidence", 1.0),
            )
        return len(facts)

    def remember(self, text: str, metadata: Optional[Dict] = None):
        """Explicitly encode something into Cortex long-term memory."""
        return self.cortex.remember(text, metadata=metadata)

    def recall_vsa(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """Direct VSA memory lookup."""
        return self.vsa.query(query, top_k=top_k)

    def recall_synaptic(self, query: str, top_k: int = 5):
        """Direct Cortex/synaptic memory lookup."""
        return self.cortex.recall(query, top_k=top_k)

    def reset_conversation(self) -> None:
        """Clear working memory to start a fresh conversation thread."""
        self.cortex.reset_working_memory()

    def sleep(self) -> Dict[str, Any]:
        """
        Run a full consolidation cycle across all memory subsystems.
        Call at end-of-session (analogous to sleep-based consolidation).
        """
        cortex_result = self.cortex.sleep()
        vsa_decayed   = self.vsa.apply_temporal_decay()
        vsa_pruned    = self.vsa.prune_weak()

        result = {
            "cortex": cortex_result,
            "vsa": {"decayed": vsa_decayed, "pruned": vsa_pruned},
            "goals": self.goals.stats,
        }
        if self.verbose:
            self._log(f"[SLEEP] {result}")
        return result

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _format_vsa_context(self, results: List[QueryResult]) -> str:
        if not results:
            return ""
        lines = [
            f"[HDC:{r.similarity:.2f}×{r.weight:.1f}] {r.label}"
            for r in results
        ]
        return "## Associative HDC Memory\n" + "\n".join(lines)

    def _log(self, msg: str) -> None:
        print(f"[Aegis] {msg}")

    # ─────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "turns": self._turn_count,
            "attention": self.attention.stats,
            "gateway":   self.gateway.stats,
            "vsa":       self.vsa.stats,
            "monitor":   self.monitor.stats,
            "cortex":    self.cortex.stats,
            "valence":   self.valence.stats,
            "goals":     self.goals.stats,
            "meta":      self.meta.stats,
        }

    def __repr__(self) -> str:
        return (
            f"Aegis-1("
            f"turns={self._turn_count}, "
            f"active_goals={len(self.goals)}, "
            f"vsa_traces={len(self.vsa)}, "
            f"facts={self.monitor.stats['knowledge_facts']}, "
            f"synapses={self.cortex.stats['long_term']['total_synapses']})"
        )