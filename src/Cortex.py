"""
Cortex: The Cognitive Middleware Layer.

Wraps any LLM with:
- Synaptic memory injection (retrieved context injected into prompt)
- Automatic memory encoding of conversations
- Neurotrophic maintenance cycles
- Working memory (short-term context window management)
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from Memory import SynapticMemory
from Synapse import Synapse
from Neurotrophic import NeurotrophicEngine


class WorkingMemory:
    """
    Short-term working memory — the cognitive 'scratchpad'.
    Holds the recent conversation context and fades over turns.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self._buffer: List[Dict[str, str]] = []

    def push(self, role: str, content: str) -> None:
        self._buffer.append({"role": role, "content": content, "ts": time.time()})
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def get_recent(self, n: int = 5) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self._buffer[-n:]]

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)


class Cortex:
    """
    The main cognitive middleware. Plug this in front of any LLM.

    Example:
        def my_llm(messages, system):
            # call OpenAI / Anthropic / Ollama / etc.
            return response_text

        def my_embedder(text):
            # call your embedding model
            return [0.1, 0.2, ...]

        brain = Cortex(llm_fn=my_llm, embed_fn=my_embedder)
        response = brain.think("What do we know about climate change?")
    """

    def __init__(
        self,
        llm_fn: Callable[[List[Dict], str], str],
        embed_fn: Callable[[str], List[float]],
        memory_path: Optional[str] = None,
        working_memory_capacity: int = 10,
        recall_top_k: int = 5,
        recall_threshold: float = 0.25,
        system_prompt: str = "You are a helpful assistant with cognitive memory.",
        auto_encode: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            llm_fn: Callable(messages: list, system: str) -> str
                    Takes a list of {"role": ..., "content": ...} and returns text.
            embed_fn: Callable(text: str) -> List[float]
                    Converts text to a semantic embedding vector.
            memory_path: Path to persist synaptic memory across sessions.
            working_memory_capacity: Max messages in short-term memory.
            recall_top_k: How many long-term memories to retrieve per turn.
            recall_threshold: Minimum similarity to surface a memory.
            system_prompt: Base system prompt for the LLM.
            auto_encode: Automatically encode each exchange into long-term memory.
            verbose: Print cognitive state on each turn.
        """
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt
        self.auto_encode = auto_encode
        self.verbose = verbose
        self.recall_top_k = recall_top_k

        # Cognitive subsystems
        self.long_term = SynapticMemory(
            embed_fn=embed_fn,
            persistence_path=memory_path,
            recall_threshold=recall_threshold,
        )
        self.working = WorkingMemory(capacity=working_memory_capacity)
        self._turn_count = 0
        self._total_tokens_saved = 0

    def think(
        self,
        user_input: str,
        extra_context: Optional[str] = None,
        recall_top_k: Optional[int] = None,
    ) -> str:
        """
        The main cognitive loop:
        1. Retrieve relevant long-term memories
        2. Build enriched system prompt
        3. Call LLM with working memory + retrieved context
        4. Encode the exchange into long-term memory
        5. Return response

        Args:
            user_input: The user's message.
            extra_context: Optional additional context to inject.
            recall_top_k: Override the default recall count.

        Returns:
            The LLM's response string.
        """
        self._turn_count += 1
        top_k = recall_top_k or self.recall_top_k

        # ── Step 1: Retrieve relevant long-term memories ──
        memories = self.long_term.recall(user_input, top_k=top_k)
        memory_context = self._format_memories(memories)

        # ── Step 2: Build enriched system prompt ──
        system = self._build_system_prompt(memory_context, extra_context)

        # ── Step 3: Add user turn to working memory ──
        self.working.push("user", user_input)

        # ── Step 4: Call LLM ──
        messages = self.working.get_recent()
        if self.verbose:
            self._log_state(user_input, memories, system)

        response = self.llm_fn(messages, system)

        # ── Step 5: Push response to working memory ──
        self.working.push("assistant", response)

        # ── Step 6: Encode exchange into long-term memory ──
        if self.auto_encode:
            self._encode_exchange(user_input, response)

        return response

    def remember(self, text: str, metadata: Optional[Dict] = None) -> Synapse:
        """
        Explicitly encode something into long-term memory.
        Use this for important facts, user preferences, or domain knowledge.
        """
        return self.long_term.encode(text, metadata=metadata)

    def recall(self, query: str, top_k: int = 5) -> List[Tuple[Synapse, float]]:
        """
        Directly query long-term memory. Useful for inspection.
        """
        return self.long_term.recall(query, top_k=top_k)

    def forget(self, synapse_id: str) -> bool:
        """Remove a specific memory by ID."""
        return self.long_term.forget(synapse_id)

    def sleep(self) -> Dict[str, Any]:
        """
        Run a full neurotrophic maintenance cycle.
        Call this periodically (e.g., end of session) to consolidate memories.
        Analogous to sleep-based memory consolidation in the brain.
        """
        result = self.long_term.housekeeping()
        if self.verbose:
            print(f"\n[SLEEP CYCLE] {result}")
        return result

    def reset_working_memory(self) -> None:
        """Clear short-term working memory (start a new conversation thread)."""
        self.working.clear()

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _format_memories(self, memories: List[Tuple[Synapse, float]]) -> str:
        if not memories:
            return ""
        lines = []
        for synapse, score in memories:
            strength = "★★★" if synapse.weight > 3 else "★★" if synapse.weight > 1.5 else "★"
            lines.append(f"{strength} [{score:.2f}] {synapse.trace}")
        return "\n".join(lines)

    def _build_system_prompt(
        self,
        memory_context: str,
        extra_context: Optional[str],
    ) -> str:
        parts = [self.system_prompt]

        if memory_context:
            parts.append(
                "\n\n## Relevant Long-Term Memories\n"
                "The following memories are retrieved from your cognitive store, "
                "ranked by relevance × synaptic strength:\n"
                + memory_context
            )

        if extra_context:
            parts.append(f"\n\n## Additional Context\n{extra_context}")

        return "\n".join(parts)

    def _encode_exchange(self, user_input: str, response: str) -> None:
        """
        Encode the conversation exchange as a memory.
        Compresses exchange into a single trace for efficiency.
        """
        trace = f"Q: {user_input[:200]}\nA: {response[:400]}"
        self.long_term.encode(
            trace,
            metadata={
                "type": "conversation",
                "turn": self._turn_count,
                "encoded_at": time.time(),
            },
        )

    def _log_state(
        self,
        user_input: str,
        memories: List[Tuple[Synapse, float]],
        system: str,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"[CORTEX TURN {self._turn_count}]")
        print(f"  Input: {user_input[:80]}...")
        print(f"  Memories retrieved: {len(memories)}")
        for s, score in memories:
            print(f"    • [{score:.3f}] {s.trace[:60]}...")
        print(f"  Long-term synapses: {len(self.long_term)}")
        print(f"  Working memory turns: {len(self.working)}")
        print(f"{'='*60}")

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "turns": self._turn_count,
            "working_memory_size": len(self.working),
            "long_term": self.long_term.stats,
        }