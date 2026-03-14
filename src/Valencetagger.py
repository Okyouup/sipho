"""
EmotionalValenceTagger: Amygdala-Inspired Affective Tagging.

The amygdala attaches emotional significance to stimuli, influencing
how strongly memories are encoded and how urgently they are processed.
High-arousal negative events (fear, anger, crisis) are remembered better
than neutral ones — this is *affective memory consolidation*.

Responsibilities:
- Assign an emotional valence score to any text: [-1.0 (negative) → +1.0 (positive)]
- Assign an arousal score: [0.0 (calm) → 1.0 (highly activated)]
- Label the dominant emotion (joy, anger, fear, sadness, surprise, disgust, neutral)
- Provide a memory_weight multiplier so high-arousal events are encoded more strongly
- Tag metadata on Synapse objects when memories are encoded

Requirements:
    No external dependencies (stdlib only)
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

class Emotion(Enum):
    JOY      = "joy"
    TRUST    = "trust"
    FEAR     = "fear"
    SURPRISE = "surprise"
    SADNESS  = "sadness"
    DISGUST  = "disgust"
    ANGER    = "anger"
    NEUTRAL  = "neutral"


@dataclass
class ValenceTag:
    """Emotional assessment of a piece of text."""
    valence: float              # [-1.0, +1.0] — negative to positive
    arousal: float              # [0.0, 1.0]   — calm to highly activated
    dominant_emotion: Emotion
    emotion_scores: Dict[str, float]  # Raw per-emotion scores
    memory_weight: float        # Multiplier for Synapse encoding strength
    text_preview: str           # First 80 chars of tagged text
    latency_ms: float = 0.0

    @property
    def is_negative(self) -> bool:
        return self.valence < -0.2

    @property
    def is_positive(self) -> bool:
        return self.valence > 0.2

    @property
    def is_high_arousal(self) -> bool:
        return self.arousal > 0.6

    def to_metadata(self) -> Dict:
        """Pack into a dict suitable for Synapse.metadata."""
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "emotion": self.dominant_emotion.value,
            "memory_weight_boost": round(self.memory_weight, 4),
        }

    def __repr__(self) -> str:
        return (
            f"ValenceTag(emotion={self.dominant_emotion.value}, "
            f"valence={self.valence:+.2f}, arousal={self.arousal:.2f}, "
            f"weight×{self.memory_weight:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EmotionalValenceTagger
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalValenceTagger:
    """
    Amygdala analogue: attaches affective significance to text.

    Uses a rule-based keyword lexicon (no external deps).
    For production use, replace _score_emotions() with a fine-tuned
    sentiment model (e.g. cardiffnlp/twitter-roberta-base-emotion).

    Memory weight multiplier logic:
        - High-arousal negative (fear, anger, crisis): 1.8×   — encode strongly
        - High-arousal positive (joy, excitement):     1.4×   — encode well
        - Neutral / low arousal:                       1.0×   — encode normally
        - Very low salience:                           0.7×   — weak encoding

    Usage:
        tagger = EmotionalValenceTagger()
        tag = tagger.tag("The production server just crashed!")
        synapse = memory.encode(text, metadata=tag.to_metadata())
    """

    # Lexicon: (pattern, valence_contribution, arousal_contribution, emotion)
    # valence: positive = pleasant, negative = unpleasant
    # arousal: high = activated/energised, low = calm/depressed
    _LEXICON: List[Tuple[str, float, float, Emotion]] = [
        # ── Joy / positive ──
        (r"\b(happy|joy|joyful|delighted|excited|thrilled|love|wonderful|amazing|great|excellent|fantastic|brilliant|perfect|celebrate|congratulations)\b",
         +0.80, 0.70, Emotion.JOY),
        (r"\b(good|nice|pleased|glad|enjoy|fun|pleasant|satisfying|helpful|useful|correct|right|success|solved|works|working)\b",
         +0.45, 0.35, Emotion.JOY),

        # ── Trust / calm positive ──
        (r"\b(trust|reliable|safe|secure|stable|confident|sure|clear|understood|agree|correct|confirmed|verified)\b",
         +0.40, 0.20, Emotion.TRUST),

        # ── Surprise (can be positive or negative — slightly positive here) ──
        (r"\b(surprised|unexpected|suddenly|wait|wow|whoa|really|seriously|actually|interesting|fascinating|strange|odd|unusual)\b",
         +0.10, 0.65, Emotion.SURPRISE),

        # ── Fear / anxiety ──
        (r"\b(afraid|scared|frightened|terrified|panic|anxious|worried|nervous|dread|threat|danger|risk|unsafe|vulnerable)\b",
         -0.75, 0.85, Emotion.FEAR),
        (r"\b(urgent|emergency|critical|alert|warning|immediately|asap|now|quickly)\b",
         -0.30, 0.90, Emotion.FEAR),

        # ── Anger ──
        (r"\b(angry|anger|furious|rage|outraged|frustrated|irritated|annoyed|hate|loathe|unacceptable|ridiculous|absurd|stupid)\b",
         -0.80, 0.90, Emotion.ANGER),
        (r"\b(wrong|broken|fail|failed|failure|bug|error|crash|not working|doesn\'t work|still broken)\b",
         -0.40, 0.65, Emotion.ANGER),

        # ── Sadness ──
        (r"\b(sad|unhappy|depressed|miserable|hopeless|disappointed|regret|sorry|unfortunate|lost|miss|grief|mourn)\b",
         -0.70, 0.40, Emotion.SADNESS),
        (r"\b(problem|issue|concern|trouble|difficult|hard|struggling|stuck|confused|lost)\b",
         -0.25, 0.35, Emotion.SADNESS),

        # ── Disgust ──
        (r"\b(disgusting|horrible|awful|terrible|dreadful|appalling|revolting|nasty|worst|garbage|trash|useless|worthless)\b",
         -0.85, 0.70, Emotion.DISGUST),
    ]

    def __init__(
        self,
        high_arousal_neg_boost: float = 1.8,
        high_arousal_pos_boost: float = 1.4,
        low_arousal_penalty:    float = 0.7,
        verbose: bool = False,
    ):
        """
        Args:
            high_arousal_neg_boost: Memory weight multiplier for intense negative events.
            high_arousal_pos_boost: Memory weight multiplier for intense positive events.
            low_arousal_penalty: Memory weight multiplier for neutral/flat content.
            verbose: Log tag decisions.
        """
        self.high_arousal_neg_boost = high_arousal_neg_boost
        self.high_arousal_pos_boost = high_arousal_pos_boost
        self.low_arousal_penalty    = low_arousal_penalty
        self.verbose                = verbose
        self._tag_count = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def tag(self, text: str) -> ValenceTag:
        """
        Tag text with emotional valence, arousal, and memory weight.

        Args:
            text: Any string (user input, LLM response, memory trace).

        Returns:
            ValenceTag with affective metadata.
        """
        t0 = time.perf_counter()
        self._tag_count += 1

        # Score each emotion
        emotion_scores = self._score_emotions(text)

        # Compute aggregate valence and arousal
        valence, arousal = self._aggregate(text, emotion_scores)

        # Dominant emotion
        dominant = max(emotion_scores, key=emotion_scores.get)
        dominant_emotion = Emotion(dominant)

        # Memory weight multiplier
        memory_weight = self._compute_memory_weight(valence, arousal)

        tag = ValenceTag(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominant_emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            memory_weight=round(memory_weight, 4),
            text_preview=text[:80],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

        if self.verbose:
            logger.debug(f"[Valence] {tag}")

        return tag

    def tag_batch(self, texts: List[str]) -> List[ValenceTag]:
        return [self.tag(t) for t in texts]

    # ─────────────────────────────────────────────
    # Scoring Internals
    # ─────────────────────────────────────────────

    def _score_emotions(self, text: str) -> Dict[str, float]:
        """
        Score each emotion category based on lexicon matches.
        Returns a dict: {emotion_value: score}.
        """
        counts: Dict[str, float] = {e.value: 0.0 for e in Emotion}
        text_lower = text.lower()

        for pattern, valence_contrib, arousal_contrib, emotion in self._LEXICON:
            matches = re.findall(pattern, text_lower, re.I)
            if matches:
                # Score is proportional to number of matches (capped)
                hit_strength = min(len(matches) * 0.5, 1.5)
                counts[emotion.value] += hit_strength

        # Neutral score is inversely proportional to total emotional content
        total_emotion = sum(v for k, v in counts.items() if k != Emotion.NEUTRAL.value)
        counts[Emotion.NEUTRAL.value] = max(0.0, 1.5 - total_emotion)

        # Normalize
        total = sum(counts.values())
        if total > 0:
            counts = {k: v / total for k, v in counts.items()}

        return counts

    def _aggregate(self, text: str, emotion_scores: Dict[str, float]) -> Tuple[float, float]:
        """Derive aggregate valence [-1, 1] and arousal [0, 1] from scores."""
        # Valence contributions per emotion
        _VALENCE_MAP = {
            Emotion.JOY.value:      +0.85,
            Emotion.TRUST.value:    +0.50,
            Emotion.SURPRISE.value: +0.10,
            Emotion.NEUTRAL.value:   0.00,
            Emotion.SADNESS.value:  -0.70,
            Emotion.DISGUST.value:  -0.80,
            Emotion.FEAR.value:     -0.75,
            Emotion.ANGER.value:    -0.80,
        }
        _AROUSAL_MAP = {
            Emotion.JOY.value:      0.65,
            Emotion.TRUST.value:    0.20,
            Emotion.SURPRISE.value: 0.75,
            Emotion.NEUTRAL.value:  0.10,
            Emotion.SADNESS.value:  0.35,
            Emotion.DISGUST.value:  0.65,
            Emotion.FEAR.value:     0.85,
            Emotion.ANGER.value:    0.90,
        }

        valence = sum(emotion_scores.get(e, 0.0) * v for e, v in _VALENCE_MAP.items())
        arousal = sum(emotion_scores.get(e, 0.0) * a for e, a in _AROUSAL_MAP.items())

        # Clamp
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        # Boost arousal for exclamation marks / caps
        if len(re.findall(r"[!]", text)) >= 2:
            arousal = min(arousal + 0.15, 1.0)
        if len(re.findall(r"[A-Z]{3,}", text)) >= 1:
            arousal = min(arousal + 0.10, 1.0)

        return valence, arousal

    def _compute_memory_weight(self, valence: float, arousal: float) -> float:
        """
        High arousal → stronger encoding (matches biological amygdala effect).
        Negative high-arousal events get the biggest boost (survival relevance).
        """
        if arousal > 0.6 and valence < -0.2:
            return self.high_arousal_neg_boost
        if arousal > 0.6 and valence > 0.2:
            return self.high_arousal_pos_boost
        if arousal < 0.25:
            return self.low_arousal_penalty
        # Linear interpolation for mid-range arousal
        return 1.0 + (arousal - 0.25) * 0.5

    @property
    def stats(self) -> Dict:
        return {"total_tagged": self._tag_count}

    def __repr__(self) -> str:
        return f"EmotionalValenceTagger(tags={self._tag_count})"