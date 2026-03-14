"""
ExecutiveMonitor: Symbolic Conflict Monitor & Hallucination Interceptor (Phase 3 of Aegis-1).

Inspired by the brain's Anterior Cingulate Cortex (ACC), which monitors
for response conflicts and signals the need for cognitive control.

Responsibilities:
- Maintain a local Symbolic Knowledge Graph (SKG)
- Validate LLM output against stored facts (conflict detection)
- Issue inhibitory signals when contradictions are detected
- Provide a "re-think" prompt that guides the LLM to self-correct
- Track validation history for audit trails

The monitor is intentionally conservative: it only flags *definite*
contradictions backed by stored facts, not mere uncertainty.

Requirements:
    pip install numpy
"""

import re
import math
import time
import logging
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class ConflictSeverity(Enum):
    NONE     = "none"       # No conflict detected
    WEAK     = "weak"       # Soft contradiction (possible hallucination)
    MODERATE = "moderate"   # Likely contradiction
    STRONG   = "strong"     # Definite contradiction — inhibit and rethink


@dataclass
class Conflict:
    """A detected conflict between LLM output and a stored fact."""
    llm_claim: str
    stored_fact: str
    severity: ConflictSeverity
    confidence: float       # [0, 1] — how certain we are of the conflict
    fact_id: str
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "llm_claim": self.llm_claim,
            "stored_fact": self.stored_fact,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "fact_id": self.fact_id,
        }


@dataclass
class ValidationResult:
    """Output from the ExecutiveMonitor for a single LLM response."""
    passed: bool                        # True if no strong conflicts
    output: str                         # The original LLM output
    conflicts: List[Conflict]           # All detected conflicts
    rethink_prompt: Optional[str]       # Injected prompt if failed
    severity: ConflictSeverity          # Worst severity found
    checked_facts: int                  # How many facts were checked
    latency_ms: float = 0.0

    @property
    def requires_rethink(self) -> bool:
        return self.severity in (ConflictSeverity.MODERATE, ConflictSeverity.STRONG)

    def __repr__(self) -> str:
        return (
            f"ValidationResult(passed={self.passed}, "
            f"severity={self.severity.value}, "
            f"conflicts={len(self.conflicts)})"
        )


@dataclass
class KnowledgeFact:
    """A single ground-truth fact in the Symbolic Knowledge Graph."""
    id: str
    text: str                   # The fact in natural language
    category: str               # e.g. "physics", "user_preference", "world_fact"
    negation: Optional[str]     # Auto-derived negation for conflict detection
    keywords: List[str]         # Extracted keywords for fast pre-filtering
    confidence: float = 1.0     # How certain we are this fact is true
    source: str = "manual"
    created_at: float = field(default_factory=time.time)
    check_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "negation": self.negation,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "KnowledgeFact":
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class SymbolicKnowledgeGraph:
    """
    Lightweight in-process knowledge store.

    Uses keyword indexing for O(1) candidate selection before
    running deeper semantic checks.
    """

    _NEGATION_PREFIXES = [
        ("is ", "is not "),
        ("are ", "are not "),
        ("was ", "was not "),
        ("has ", "has not "),
        ("can ", "cannot "),
        ("will ", "will not "),
        ("does ", "does not "),
        ("did ", "did not "),
    ]

    def __init__(self):
        self._facts: Dict[str, KnowledgeFact] = {}
        self._keyword_index: Dict[str, List[str]] = {}  # keyword → [fact_ids]
        self._fact_counter = 0

    def add(
        self,
        text: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> KnowledgeFact:
        """Add a fact to the knowledge graph."""
        fact_id = f"fact_{self._fact_counter:05d}"
        self._fact_counter += 1
        keywords = self._extract_keywords(text)
        negation = self._derive_negation(text)

        fact = KnowledgeFact(
            id=fact_id,
            text=text,
            category=category,
            negation=negation,
            keywords=keywords,
            confidence=confidence,
            source=source,
        )
        self._facts[fact_id] = fact
        for kw in keywords:
            self._keyword_index.setdefault(kw, []).append(fact_id)

        return fact

    def remove(self, fact_id: str) -> bool:
        if fact_id not in self._facts:
            return False
        for kw in self._facts[fact_id].keywords:
            ids = self._keyword_index.get(kw, [])
            if fact_id in ids:
                ids.remove(fact_id)
        del self._facts[fact_id]
        return True

    def get_candidates(self, text: str, max_candidates: int = 20) -> List[KnowledgeFact]:
        """Fast keyword-based candidate selection."""
        keywords = self._extract_keywords(text)
        candidate_ids: Dict[str, int] = {}
        for kw in keywords:
            for fid in self._keyword_index.get(kw, []):
                candidate_ids[fid] = candidate_ids.get(fid, 0) + 1

        # Sort by keyword overlap count
        sorted_ids = sorted(candidate_ids, key=lambda x: -candidate_ids[x])
        return [self._facts[fid] for fid in sorted_ids[:max_candidates]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {"facts": {fid: f.to_dict() for fid, f in self._facts.items()}}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> int:
        with open(path, "r") as f:
            data = json.load(f)
        for fid, fd in data.get("facts", {}).items():
            fact = KnowledgeFact.from_dict(fd)
            self._facts[fid] = fact
            for kw in fact.keywords:
                self._keyword_index.setdefault(kw, []).append(fid)
        return len(self._facts)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract meaningful keywords (strip stopwords)."""
        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "that", "this",
            "it", "its", "and", "or", "but", "not", "no",
        }
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return list(set(t for t in tokens if t not in _STOPWORDS))

    @staticmethod
    def _derive_negation(text: str) -> Optional[str]:
        """Simple rule-based negation for conflict detection."""
        lower = text.lower().strip()
        for pos, neg in SymbolicKnowledgeGraph._NEGATION_PREFIXES:
            if pos in lower:
                return lower.replace(pos, neg, 1)
            if neg in lower:
                return lower.replace(neg, pos, 1)
        return None

    def __len__(self) -> int:
        return len(self._facts)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Conflict Checker
# ─────────────────────────────────────────────────────────────────────────────

def _token_overlap_score(a: str, b: str) -> float:
    """
    Jaccard similarity on word sets.
    Used as a fast proxy for semantic similarity when no embedder is available.
    """
    a_tokens = set(re.findall(r"\b\w+\b", a.lower()))
    b_tokens = set(re.findall(r"\b\w+\b", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union > 0 else 0.0


def _negation_conflict_score(claim: str, fact: str) -> float:
    """
    Check if claim is semantically close to the *negation* of the fact.

    A high score means the claim likely contradicts the fact.
    """
    # Direct negation patterns in claim
    negation_markers = [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bcannot\b",
        r"\bwon't\b", r"\bisn't\b", r"\baren't\b", r"\bwasn't\b",
        r"\bweren't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bdon't\b",
    ]
    claim_negated = any(re.search(p, claim, re.I) for p in negation_markers)
    fact_negated  = any(re.search(p, fact,  re.I) for p in negation_markers)

    # Base overlap between claim and fact
    overlap = _token_overlap_score(claim, fact)

    # If one is negated and the other isn't, and they share tokens → conflict
    if claim_negated != fact_negated and overlap > 0.3:
        return overlap * 0.8

    return 0.0


def _numerical_conflict_score(claim: str, fact: str) -> float:
    """
    Detect contradictory numerical claims.
    e.g. "The year is 2022" vs stored "The year is 2024"
    """
    nums_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
    nums_fact  = re.findall(r"\b\d+(?:\.\d+)?\b", fact)

    if not nums_claim or not nums_fact:
        return 0.0

    # Check if same context words but different numbers
    context_overlap = _token_overlap_score(
        re.sub(r"\d+", "", claim),
        re.sub(r"\d+", "", fact),
    )
    if context_overlap < 0.4:
        return 0.0

    # If numbers differ, it's a potential conflict
    if set(nums_claim) != set(nums_fact):
        return 0.6 * context_overlap

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Executive Monitor
# ─────────────────────────────────────────────────────────────────────────────

class ExecutiveMonitor:
    """
    Anterior Cingulate Cortex analogue for Aegis-1.

    Monitors LLM outputs for conflicts with stored symbolic knowledge.
    Issues an inhibitory "re-think" prompt when contradictions are detected.

    Usage:
        monitor = ExecutiveMonitor()
        monitor.learn("Water boils at 100°C at sea level.", category="physics")
        monitor.learn("The capital of France is Paris.", category="geography")

        result = monitor.validate("The capital of France is Lyon.")
        if result.requires_rethink:
            corrected = llm(result.rethink_prompt)
    """

    _SEVERITY_THRESHOLDS = {
        ConflictSeverity.WEAK:     0.25,
        ConflictSeverity.MODERATE: 0.50,
        ConflictSeverity.STRONG:   0.72,
    }

    def __init__(
        self,
        knowledge_path: Optional[str] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        strong_threshold: float = 0.72,
        moderate_threshold: float = 0.50,
        weak_threshold: float = 0.25,
        max_rethink_attempts: int = 3,
        verbose: bool = False,
    ):
        """
        Args:
            knowledge_path: Optional path to persist the knowledge graph.
            embed_fn: Optional semantic embedder for richer conflict detection.
            strong_threshold: Score above which a STRONG conflict is flagged.
            moderate_threshold: Score for MODERATE conflict.
            weak_threshold: Score for WEAK conflict.
            max_rethink_attempts: Max times to re-prompt the LLM on failure.
            verbose: Log validation details.
        """
        self.embed_fn = embed_fn
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.weak_threshold = weak_threshold
        self.max_rethink_attempts = max_rethink_attempts
        self.verbose = verbose

        self.skg = SymbolicKnowledgeGraph()
        self._validation_log: List[Dict] = []
        self._rethink_count = 0
        self._pass_count = 0

        if knowledge_path and os.path.exists(knowledge_path):
            self.skg.load(knowledge_path)
            logger.info(f"[Monitor] Loaded {len(self.skg)} facts from {knowledge_path}")

        self._knowledge_path = knowledge_path

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def learn(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "manual",
    ) -> KnowledgeFact:
        """
        Add a ground-truth fact to the knowledge graph.

        Args:
            fact: Natural language statement of truth.
            category: Domain label (e.g. "physics", "user_prefs").
            confidence: How certain this fact is [0, 1].
            source: Origin of the fact.
        """
        stored = self.skg.add(fact, category=category, confidence=confidence, source=source)
        if self._knowledge_path:
            self.skg.save(self._knowledge_path)
        if self.verbose:
            logger.debug(f"[Monitor] Learned fact: {fact[:60]}... [{category}]")
        return stored

    def learn_batch(self, facts: List[Dict]) -> int:
        """
        Batch-load facts from a list of dicts.

        Each dict must have "text"; optionally "category", "confidence", "source".

        Returns:
            Number of facts added.
        """
        for f in facts:
            self.learn(
                fact=f["text"],
                category=f.get("category", "general"),
                confidence=f.get("confidence", 1.0),
                source=f.get("source", "batch"),
            )
        return len(facts)

    def validate(
        self,
        llm_output: str,
        context: Optional[str] = None,
    ) -> ValidationResult:
        """
        Check an LLM output for conflicts with stored knowledge.

        Args:
            llm_output: The raw text generated by the LLM.
            context: Optional original user query for richer rethink prompts.

        Returns:
            ValidationResult — check `.requires_rethink` and `.rethink_prompt`.
        """
        t0 = time.perf_counter()

        if not self.skg or len(self.skg) == 0:
            # No knowledge → all outputs pass trivially
            self._pass_count += 1
            return ValidationResult(
                passed=True,
                output=llm_output,
                conflicts=[],
                rethink_prompt=None,
                severity=ConflictSeverity.NONE,
                checked_facts=0,
                latency_ms=0.0,
            )

        # Split output into sentences for granular checking
        sentences = self._split_sentences(llm_output)
        all_conflicts: List[Conflict] = []

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            candidates = self.skg.get_candidates(sentence, max_candidates=15)
            for fact in candidates:
                fact.check_count += 1
                conflict = self._check_conflict(sentence, fact)
                if conflict:
                    all_conflicts.append(conflict)

        # Determine worst severity
        worst = ConflictSeverity.NONE
        for c in all_conflicts:
            if c.severity == ConflictSeverity.STRONG:
                worst = ConflictSeverity.STRONG
                break
            if c.severity == ConflictSeverity.MODERATE:
                worst = ConflictSeverity.MODERATE
            elif worst == ConflictSeverity.NONE:
                worst = ConflictSeverity.WEAK

        passed = worst not in (ConflictSeverity.MODERATE, ConflictSeverity.STRONG)

        rethink_prompt = None
        if not passed:
            rethink_prompt = self._build_rethink_prompt(
                llm_output, all_conflicts, context
            )
            self._rethink_count += 1
        else:
            self._pass_count += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = ValidationResult(
            passed=passed,
            output=llm_output,
            conflicts=all_conflicts,
            rethink_prompt=rethink_prompt,
            severity=worst,
            checked_facts=len(self.skg),
            latency_ms=latency_ms,
        )

        self._validation_log.append({
            "output_preview": llm_output[:80],
            "severity": worst.value,
            "conflicts": len(all_conflicts),
            "passed": passed,
            "timestamp": time.time(),
        })

        if self.verbose:
            logger.info(f"[Monitor] {result}")

        return result

    def forget(self, fact_id: str) -> bool:
        """Remove a fact from the knowledge graph."""
        return self.skg.remove(fact_id)

    def audit_log(self, last_n: int = 10) -> List[Dict]:
        """Return the most recent validation log entries."""
        return self._validation_log[-last_n:]

    # ─────────────────────────────────────────────
    # Conflict Detection
    # ─────────────────────────────────────────────

    def _check_conflict(
        self, claim: str, fact: KnowledgeFact
    ) -> Optional[Conflict]:
        """
        Run all conflict detection heuristics against a single fact.
        Returns a Conflict if detected, else None.
        """
        scores: List[float] = []

        # 1. Negation conflict
        neg_score = _negation_conflict_score(claim, fact.text)
        scores.append(neg_score)

        # 2. Numerical contradiction
        num_score = _numerical_conflict_score(claim, fact.text)
        scores.append(num_score)

        # 3. Semantic embedder conflict (if available)
        if self.embed_fn and fact.negation:
            try:
                claim_vec = np.array(self.embed_fn(claim))
                neg_vec   = np.array(self.embed_fn(fact.negation))
                fact_vec  = np.array(self.embed_fn(fact.text))

                # High similarity to negation + high similarity to fact context
                neg_sim  = self._cosine(claim_vec, neg_vec)
                fact_sim = self._cosine(claim_vec, fact_vec)

                if neg_sim > 0.6 and fact_sim > 0.4:
                    embed_score = (neg_sim + fact_sim) / 2.0
                    scores.append(embed_score * 0.9)
            except Exception:
                pass

        composite_score = max(scores) * fact.confidence if scores else 0.0

        if composite_score < self.weak_threshold:
            return None

        if composite_score >= self.strong_threshold:
            severity = ConflictSeverity.STRONG
        elif composite_score >= self.moderate_threshold:
            severity = ConflictSeverity.MODERATE
        else:
            severity = ConflictSeverity.WEAK

        return Conflict(
            llm_claim=claim,
            stored_fact=fact.text,
            severity=severity,
            confidence=composite_score,
            fact_id=fact.id,
        )

    # ─────────────────────────────────────────────
    # Re-Think Prompt Builder
    # ─────────────────────────────────────────────

    def _build_rethink_prompt(
        self,
        llm_output: str,
        conflicts: List[Conflict],
        context: Optional[str],
    ) -> str:
        """
        Build an inhibitory re-think prompt for the LLM.
        Injects the conflicting facts so the LLM can self-correct.
        """
        strong_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.STRONG]
        moderate_conflicts = [c for c in conflicts if c.severity == ConflictSeverity.MODERATE]
        flagged = (strong_conflicts + moderate_conflicts)[:3]

        fact_lines = "\n".join(
            f"  - Fact: \"{c.stored_fact}\" "
            f"[conflicts with: \"{c.llm_claim[:80]}\"]"
            for c in flagged
        )

        context_line = f"\nOriginal query: \"{context}\"\n" if context else ""

        return (
            "⚠️  Your previous response contained information that conflicts with "
            "verified facts in the knowledge base. Please revise your response.\n"
            f"{context_line}"
            f"\nKnown conflicts:\n{fact_lines}\n\n"
            "Please generate a corrected response that is consistent with these facts. "
            "Do not repeat the conflicting claims."
        )

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences for granular checking."""
        # Simple regex-based splitter
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    @property
    def stats(self) -> Dict:
        total = self._pass_count + self._rethink_count
        return {
            "total_validations": total,
            "pass_count": self._pass_count,
            "rethink_count": self._rethink_count,
            "rethink_rate": round(self._rethink_count / total, 4) if total else 0.0,
            "knowledge_facts": len(self.skg),
        }