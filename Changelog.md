# Changelog

All notable changes to Aegis-1 are documented here.

---

## [2.0.0] — 2026-03-16

### Bug Fix

**`VSAMemory.py` — startup crash restored**

The file was accidentally overwritten with `Valencetagger` content during the
`Vsamemory.py` → `VSAMemory.py` rename. Python found the file but could not
import `VSAMemory` from it, producing the error:

```
cannot import name 'VSAMemory' from 'VSAMemory'
```

The full HDC memory kernel (`VSAMemory`, `HDCodebook`, `HDMemoryTrace`,
`QueryResult`, and all HDC primitives) has been restored. No logic was changed
— this is a pure file-content fix.

---

### Improved: `Valencetagger.py`

**Problem:** The original used `re.findall` to count keyword matches with no
awareness of surrounding context. `"not angry"` scored the same as `"angry"`.
`"terrible pun, lol"` registered as genuine disgust. Every emotion keyword
fired at full weight regardless of whether it was negated, softened, or clearly
part of a joke.

**Changes:**

- **Negation scope detection.** Each keyword match now inspects a 50-character
  window before it for negation words (`not`, `never`, `isn't`, etc.). A
  negated match contributes 15% of its normal weight. Scope is bounded by
  sentence boundaries so `"I'm not angry. The crash was terrible."` does not
  suppress `terrible`.

- **Humor and sarcasm awareness.** If a humor marker (`lol`, `haha`, `jk`,
  `/s`, `"terrible pun"`, etc.) appears within 120 characters of an
  anger/disgust/fear/sadness keyword, those hits are dampened to 25%.
  `"terrible pun, lol"` now reads as near-neutral.

- **Intensity modifiers.** A 35-character window before each match checks for
  intensifiers (`very`, `extremely`, `absolutely`) → 1.5× weight, or
  diminishers (`a bit`, `slightly`, `kind of`) → 0.5× weight.

- **Fictional framing detection.** If the text contains phrases like
  `"writing a story"`, `"my fictional character"`, `"hypothetically"`,
  `"let's pretend"`, etc., every emotion hit is multiplied by 0.35. The tagger
  no longer treats creative writing prompts as genuine emotional events.

---

### Improved: `Attentionfilter.py`

**Problem:** Urgency patterns were pure regex with no negation awareness.
`"this is not an emergency"` triggered CRITICAL priority. Fictional or
hypothetical framing had no effect on urgency scoring.

**Changes:**

- **Per-keyword negation check.** Each urgency pattern match checks a bounded
  window before it for negation words. `"not an emergency"` now scores 8% of
  the urgency it would otherwise get.

- **Fictional framing detection.** If the current message or any of the last 5
  messages established a fictional/hypothetical context, urgency is multiplied
  by 0.25. A multi-turn setup like `"I'm writing a thriller"` followed by
  `"describe the emergency"` is handled correctly.

- **Sentence-boundary scoping.** Negation detection is bounded by the nearest
  preceding sentence boundary, preventing cross-sentence bleed.

---

### Improved: `Perceptualgateway.py`

**Problem:** When the Nengo library is not installed (the common case), the
fallback was a plain sigmoid: `1 / (1 + exp(-10 * (x - 0.4)))`. This is
smooth and symmetric — it does not replicate the sharp threshold and
sub-threshold suppression that the actual LIF network produces, meaning
routing behaviour differed significantly between Nengo and non-Nengo
deployments.

**Changes:**

- **Analytical LIF rate-model fallback.** The sigmoid has been replaced with a
  three-step rate-model that mirrors the Nengo network topology:
  1. Excitatory drive → LIF firing rate (closed-form formula)
  2. Feed-forward to inhibitory population → inhibitory rate
  3. Lateral inhibitory feedback onto excitatory → corrected final rate

  The closed-form LIF formula used is:
  `r = 1 / (τ_ref + τ_rc · ln(1 − 1/J))` for J > 1, else 0.

  This produces a sharp threshold and strong sub-threshold suppression
  consistent with the Nengo network, so routing behaviour is now the same
  whether or not Nengo is installed.

---

### Improved: `Goalstack.py`

**Problem:** Auto-completion detection was pure keyword substring matching.
A goal `"always cite sources"` would be marked complete any time the word
`"sources"` appeared in a response — even an unrelated one. Paraphrased
completions (e.g. the model citing references without using the word
`"sources"`) were never detected.

**Changes:**

- **Optional `embed_fn` parameter.** When an embedder is provided, goal
  embeddings are pre-computed and cached at push time.

- **Semantic completion via cosine similarity.** `check_completion()` computes
  the response embedding once per turn and compares it against all cached goal
  embeddings. A goal is considered addressed if cosine similarity ≥ 0.62.

- **Minimum dwell guard.** A goal must be active for at least 2 turns before
  semantic auto-completion can fire, preventing a newly pushed goal from
  immediately matching its own wording in the first response.

- **Keyword signals run first.** Explicit `completion_signals` still fire
  immediately when matched. Semantic detection is the fallback. Deployments
  without `embed_fn` work exactly as before.

---

### Improved: `Metacognition.py`

**Problem:** Confidence scoring rewarded surface style over content quality.
A response full of numbered lists and the word `"therefore"` scored higher
than a concise, accurate answer written in plain prose. The system could be
gamed by formatting alone. There was also no check for whether the response
actually addressed the question asked.

**Changes:**

- **Semantic query-response relevance scoring.** When `embed_fn` and
  `user_query` are provided, `evaluate()` computes cosine similarity between
  the query and the response. A response that does not address the question
  gets a 0.12 confidence penalty.

- **New `IRRELEVANT` quality flag.** Fires when semantic relevance drops below
  0.40.

- **Reduced structural pattern weight.** Structure signals now contribute at
  most 0.07 to confidence (down from 0.15). They are a minor hint, not the
  main driver.

- **Style-gaming cap.** A response cannot exceed 0.70 confidence from
  structural patterns alone. Breaking past that ceiling requires memory support
  or semantic relevance.

- **Raised `OVER_HEDGED` threshold.** The flag now fires at hedge density >
  0.45 (up from 0.35). Moderate hedging on genuinely uncertain facts is correct
  epistemic practice and should not be penalised.

---

### Wiring changes required in `Aegis.py`

Three additions are needed to activate the GoalStack and MetaCognition
improvements. Without them both modules fall back to v1 behaviour silently —
nothing breaks, but semantic completion detection and relevance scoring will
not run.

**1. Pass `embed_fn` to `GoalStack`:**
```python
# Before
self.goals = GoalStack(verbose=verbose)

# After
self.goals = GoalStack(embed_fn=embed_fn, verbose=verbose)
```

**2. Pass `embed_fn` to `MetaCognition`:**
```python
# Before
self.meta = MetaCognition(
    rethink_threshold=meta_rethink_threshold,
    optimal_response_words=meta_optimal_response_words,
    verbose=verbose,
)

# After
self.meta = MetaCognition(
    rethink_threshold=meta_rethink_threshold,
    optimal_response_words=meta_optimal_response_words,
    embed_fn=embed_fn,
    verbose=verbose,
)
```

**3. Pass `user_input` as `user_query` to `meta.evaluate()`:**
```python
# Before
meta_report = self.meta.evaluate(
    response=response_text,
    validation_passed=validation.passed,
    conflict_count=len(validation.conflicts),
    memories_retrieved=cortex_memories,
    route=route.value,
)

# After
meta_report = self.meta.evaluate(
    response=response_text,
    validation_passed=validation.passed,
    conflict_count=len(validation.conflicts),
    memories_retrieved=cortex_memories,
    route=route.value,
    user_query=user_input,
)
```

Apply the same `user_query` addition to the second `meta.evaluate()` call
inside the MetaCognition-triggered rethink block.

---

## [1.0.0] — Initial release

- Seven-phase cognitive pipeline: AttentionFilter, PerceptualGateway,
  VSAMemory, ExecutiveMonitor, ValenceTagger, GoalStack, MetaCognition
- DeepSeek LLM backend with OpenAI-compatible adapter
- Synaptic long-term memory with LTP/LTD decay
- Neurotrophic consolidation and pruning
- Interactive CLI, smoke test suite, REST API, and web UI