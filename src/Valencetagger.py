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

v2 improvements over v1:
- Negation scope detection: "not angry", "isn't broken" correctly dampen emotion scores
- Humor / sarcasm awareness: "terrible pun lol" no longer reads as genuine disgust
- Intensity modifiers: "extremely happy" scores higher than "a bit happy"
- Fictional framing: "writing a story about an emergency" gets dampened real-world weight
- Per-match context window: each lexicon hit evaluated in its local textual context

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
    valence: float
    arousal: float
    dominant_emotion: Emotion
    emotion_scores: Dict[str, float]
    memory_weight: float
    text_preview: str
    latency_ms: float = 0.0
    humor_detected: bool = False
    negation_detected: bool = False
    fictional_framing: bool = False

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
        return {
            "valence":              round(self.valence, 4),
            "arousal":              round(self.arousal, 4),
            "emotion":              self.dominant_emotion.value,
            "memory_weight_boost":  round(self.memory_weight, 4),
            "humor_detected":       self.humor_detected,
            "fictional_framing":    self.fictional_framing,
        }

    def __repr__(self) -> str:
        flags = []
        if self.humor_detected:    flags.append("humor")
        if self.negation_detected: flags.append("negated")
        if self.fictional_framing: flags.append("fiction")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        return (
            f"ValenceTag(emotion={self.dominant_emotion.value}, "
            f"valence={self.valence:+.2f}, arousal={self.arousal:.2f}, "
            f"weight×{self.memory_weight:.2f}{flag_str})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EmotionalValenceTagger
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalValenceTagger:
    """
    Amygdala analogue: attaches affective significance to text.

    v2: per-match context analysis for negation, humor, intensity, and
    fictional framing. No external dependencies.

    Memory weight multiplier logic:
        - High-arousal negative (fear, anger, crisis): 1.8×
        - High-arousal positive (joy, excitement):     1.4×
        - Neutral / low arousal:                       1.0×
        - Very low salience:                           0.7×
    """

    _LEXICON: List[Tuple[str, float, float, Emotion]] = [
        # Joy / positive
        (r"\b(happy|joy|joyful|delighted|excited|thrilled|love|wonderful|amazing|great|"
         r"excellent|fantastic|brilliant|perfect|celebrate|congratulations)\b",
         +0.80, 0.70, Emotion.JOY),
        (r"\b(good|nice|pleased|glad|enjoy|fun|pleasant|satisfying|helpful|useful|"
         r"correct|right|success|solved|works|working)\b",
         +0.45, 0.35, Emotion.JOY),
        # Trust
        (r"\b(trust|reliable|safe|secure|stable|confident|sure|clear|understood|"
         r"agree|correct|confirmed|verified)\b",
         +0.40, 0.20, Emotion.TRUST),
        # Surprise
        (r"\b(surprised|unexpected|suddenly|wait|wow|whoa|really|seriously|actually|"
         r"interesting|fascinating|strange|odd|unusual)\b",
         +0.10, 0.65, Emotion.SURPRISE),
        # Fear / anxiety
        (r"\b(afraid|scared|frightened|terrified|panic|anxious|worried|nervous|"
         r"dread|threat|danger|risk|unsafe|vulnerable)\b",
         -0.75, 0.85, Emotion.FEAR),
        (r"\b(urgent|emergency|critical|alert|warning|immediately|asap|quickly)\b",
         -0.30, 0.90, Emotion.FEAR),
        # Anger
        (r"\b(angry|anger|furious|rage|outraged|frustrated|irritated|annoyed|"
         r"hate|loathe|unacceptable|ridiculous|absurd|stupid)\b",
         -0.80, 0.90, Emotion.ANGER),
        (r"\b(wrong|broken|fail|failed|failure|bug|error|crash|not working|"
         r"doesn't work|still broken)\b",
         -0.40, 0.65, Emotion.ANGER),
        # Sadness
        (r"\b(sad|unhappy|depressed|miserable|hopeless|disappointed|regret|"
         r"sorry|unfortunate|lost|miss|grief|mourn)\b",
         -0.70, 0.40, Emotion.SADNESS),
        (r"\b(problem|issue|concern|trouble|difficult|hard|struggling|stuck|confused)\b",
         -0.25, 0.35, Emotion.SADNESS),
        # Disgust
        (r"\b(disgusting|horrible|awful|terrible|dreadful|appalling|revolting|"
         r"nasty|worst|garbage|trash|useless|worthless)\b",
         -0.85, 0.70, Emotion.DISGUST),
    ]

    _NEGATION_RE = re.compile(
        r"\b(not|no|never|isn't|aren't|doesn't|don't|won't|can't|couldn't|"
        r"wouldn't|shouldn't|hardly|barely|nothing|without|nor|neither)\b",
        re.I,
    )
    _NEGATION_WINDOW = 50

    _HUMOR_RE = re.compile(
        r"\b(lol|lmao|lmfao|rofl|haha|hehe|heh|jk|just kidding|xd)\b"
        r"|\b/s\b"
        r"|\b(terrible|awful|horrible|worst)\s+(pun|joke|humor|humour|dad joke)\b"
        r"|\b(what a|oh no|oh dear).{0,30}(lol|haha|😂|🤣)\b",
        re.I,
    )
    _HUMOR_PROXIMITY = 120

    _INTENSIFIERS_RE = re.compile(
        r"\b(very|extremely|incredibly|absolutely|completely|utterly|totally|"
        r"deeply|profoundly|overwhelmingly|insanely|ridiculously|so)\b",
        re.I,
    )
    _DIMINISHERS_RE = re.compile(
        r"\b(a bit|slightly|somewhat|kind of|kinda|sort of|barely|hardly|"
        r"a little|mildly|faintly|not very|not too|not that)\b",
        re.I,
    )
    _MODIFIER_WINDOW = 35

    _FICTION_RE = re.compile(
        r"\b(writing|wrote|write)\s+(a\s+|an\s+|the\s+)?"
        r"(story|novel|scene|script|character|fiction|narrative|play|screenplay)\b"
        r"|\b(in\s+my|in\s+the)\s+(story|novel|book|game|script|fiction|narrative)\b"
        r"|\bhypothetically\b"
        r"|\bimagine\s+if\b"
        r"|\bfor\s+(a\s+|my\s+|the\s+)?(story|novel|character|game|screenplay)\b"
        r"|\bmy\s+(fictional\s+)?character\b"
        r"|\b(fictional|fictionally|as\s+fiction|as\s+a\s+joke)\b"
        r"|\blet's\s+(say|pretend|imagine)\b"
        r"|\bwhat\s+if\s+I\b",
        re.I,
    )
    _FICTION_DAMPENER = 0.35

    def __init__(
        self,
        high_arousal_neg_boost: float = 1.8,
        high_arousal_pos_boost: float = 1.4,
        low_arousal_penalty:    float = 0.7,
        negation_multiplier:    float = 0.15,
        humor_multiplier:       float = 0.25,
        verbose: bool = False,
    ):
        self.high_arousal_neg_boost = high_arousal_neg_boost
        self.high_arousal_pos_boost = high_arousal_pos_boost
        self.low_arousal_penalty    = low_arousal_penalty
        self.negation_multiplier    = negation_multiplier
        self.humor_multiplier       = humor_multiplier
        self.verbose                = verbose
        self._tag_count             = 0

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def tag(self, text: str) -> ValenceTag:
        t0 = time.perf_counter()
        self._tag_count += 1

        text_lower      = text.lower()
        fictional       = bool(self._FICTION_RE.search(text_lower))
        humor_positions = [m.start() for m in self._HUMOR_RE.finditer(text_lower)]

        emotion_scores, negation_detected = self._score_emotions(
            text_lower, fictional, humor_positions
        )

        valence, arousal = self._aggregate(text, emotion_scores)
        dominant         = Emotion(max(emotion_scores, key=emotion_scores.get))
        memory_weight    = self._compute_memory_weight(valence, arousal)

        tag = ValenceTag(
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            dominant_emotion=dominant,
            emotion_scores=emotion_scores,
            memory_weight=round(memory_weight, 4),
            text_preview=text[:80],
            latency_ms=(time.perf_counter() - t0) * 1000,
            humor_detected=bool(humor_positions),
            negation_detected=negation_detected,
            fictional_framing=fictional,
        )
        if self.verbose:
            logger.debug(f"[Valence] {tag}")
        return tag

    def tag_batch(self, texts: List[str]) -> List["ValenceTag"]:
        return [self.tag(t) for t in texts]

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _score_emotions(
        self,
        text_lower: str,
        fictional: bool,
        humor_positions: List[int],
    ) -> Tuple[Dict[str, float], bool]:
        counts: Dict[str, float] = {e.value: 0.0 for e in Emotion}
        fiction_mult = self._FICTION_DAMPENER if fictional else 1.0
        any_negation = False

        for pattern, _val, _aro, emotion in self._LEXICON:
            for match in re.finditer(pattern, text_lower, re.I):
                mstart = match.start()

                neg_mult = self._negation_mult(text_lower, mstart)
                if neg_mult < 1.0:
                    any_negation = True

                intensity_mult = self._intensity_mult(text_lower, mstart)

                humor_mult = 1.0
                if humor_positions and emotion in (
                    Emotion.ANGER, Emotion.DISGUST, Emotion.FEAR, Emotion.SADNESS
                ):
                    if any(abs(mstart - hp) <= self._HUMOR_PROXIMITY for hp in humor_positions):
                        humor_mult = self.humor_multiplier

                counts[emotion.value] += 0.5 * neg_mult * intensity_mult * humor_mult * fiction_mult

        total_emotion = sum(v for k, v in counts.items() if k != Emotion.NEUTRAL.value)
        counts[Emotion.NEUTRAL.value] = max(0.0, 1.5 - total_emotion)

        total = sum(counts.values())
        if total > 0:
            counts = {k: v / total for k, v in counts.items()}

        return counts, any_negation

    def _aggregate(self, text: str, scores: Dict[str, float]) -> Tuple[float, float]:
        _V = {
            Emotion.JOY.value:      +0.85,
            Emotion.TRUST.value:    +0.50,
            Emotion.SURPRISE.value: +0.10,
            Emotion.NEUTRAL.value:   0.00,
            Emotion.SADNESS.value:  -0.70,
            Emotion.DISGUST.value:  -0.80,
            Emotion.FEAR.value:     -0.75,
            Emotion.ANGER.value:    -0.80,
        }
        _A = {
            Emotion.JOY.value:      0.65,
            Emotion.TRUST.value:    0.20,
            Emotion.SURPRISE.value: 0.75,
            Emotion.NEUTRAL.value:  0.10,
            Emotion.SADNESS.value:  0.35,
            Emotion.DISGUST.value:  0.65,
            Emotion.FEAR.value:     0.85,
            Emotion.ANGER.value:    0.90,
        }
        valence = max(-1.0, min(1.0, sum(scores.get(e, 0.0) * v for e, v in _V.items())))
        arousal = max(0.0,  min(1.0, sum(scores.get(e, 0.0) * a for e, a in _A.items())))

        if len(re.findall(r"[!]", text)) >= 2:
            arousal = min(arousal + 0.15, 1.0)
        if len(re.findall(r"[A-Z]{3,}", text)) >= 1:
            arousal = min(arousal + 0.10, 1.0)

        return valence, arousal

    def _compute_memory_weight(self, valence: float, arousal: float) -> float:
        if arousal > 0.6 and valence < -0.2:
            return self.high_arousal_neg_boost
        if arousal > 0.6 and valence > 0.2:
            return self.high_arousal_pos_boost
        if arousal < 0.25:
            return self.low_arousal_penalty
        return 1.0 + (arousal - 0.25) * 0.5

    def _negation_mult(self, text: str, mstart: int) -> float:
        prefix = text[max(0, mstart - self._NEGATION_WINDOW): mstart]
        last_b = max(prefix.rfind(". "), prefix.rfind("! "),
                     prefix.rfind("? "), prefix.rfind("\n"))
        if last_b != -1:
            prefix = prefix[last_b + 2:]
        return self.negation_multiplier if self._NEGATION_RE.search(prefix) else 1.0

    def _intensity_mult(self, text: str, mstart: int) -> float:
        prefix = text[max(0, mstart - self._MODIFIER_WINDOW): mstart]
        has_int = bool(self._INTENSIFIERS_RE.search(prefix))
        has_dim = bool(self._DIMINISHERS_RE.search(prefix))
        if has_int and has_dim:
            i = self._INTENSIFIERS_RE.search(prefix).start()
            d = self._DIMINISHERS_RE.search(prefix).start()
            return 1.5 if i > d else 0.5
        if has_int: return 1.5
        if has_dim: return 0.5
        return 1.0

    @property
    def stats(self) -> Dict:
        return {"total_tagged": self._tag_count}

    def __repr__(self) -> str:
        return f"EmotionalValenceTagger(tags={self._tag_count})"