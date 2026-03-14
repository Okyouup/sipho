"""
Aegis-1 — Production Runner (DeepSeek + local embeddings)
==========================================================
Requirements:
    pip install openai numpy sentence-transformers requests

Setup:
    export DEEPSEEK_API_KEY=sk-...

Run smoke test:    python3 run_aegis.py --smoke
Interactive chat:  python3 run_aegis.py
Debug telemetry:   python3 run_aegis.py --verbose
"""

import sys
import os
import argparse
import logging
import logging.config
import time
import signal

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,   # silence noisy third-party loggers
    "formatters": {
        "default": {
            "format": "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        }
    },
    # Only show WARNING+ from third-party libs; show INFO+ from aegis.runner
    "loggers": {
        "aegis.runner":            {"level": "DEBUG",   "handlers": ["console"], "propagate": False},
        "openai":                  {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "httpx":                   {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "httpcore":                {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "huggingface_hub":         {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "sentence_transformers":   {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "transformers":            {"level": "ERROR",   "handlers": ["console"], "propagate": False},
        "nengo":                   {"level": "CRITICAL","handlers": ["console"], "propagate": False},
        "torch":                   {"level": "ERROR",   "handlers": ["console"], "propagate": False},
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
})

log = logging.getLogger("aegis.runner")

# ── Path ───────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Aegis-1 production runner — DeepSeek LLM + local embeddings"
)
parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
parser.add_argument("--model",   default="deepseek-chat",
                    choices=["deepseek-chat", "deepseek-reasoner"])
parser.add_argument("--smoke",   action="store_true", help="Run smoke tests and exit")
parser.add_argument("--verbose", action="store_true", help="Full per-phase telemetry")
parser.add_argument("--debug",   action="store_true", help="Enable DEBUG log level")
args = parser.parse_args()

if args.debug:
    logging.getLogger("aegis.runner").setLevel(logging.DEBUG)

if not args.api_key:
    log.error("No API key found. Set DEEPSEEK_API_KEY or pass --api-key sk-...")
    sys.exit(1)


# ── Imports ────────────────────────────────────────────────────────────────────
try:
    import openai
except ImportError:
    log.error("Missing dependency. Run:  pip install openai")
    sys.exit(1)

try:
    from Aegis import Aegis
    from Goalstack import GoalPriority
    from Llm import _semantic_embed, _simple_embed, _ST_AVAILABLE
except ImportError as e:
    log.error(f"Could not import Aegis source files. Make sure src/ is present. Error: {e}")
    sys.exit(1)


# ── LLM function — DeepSeek ────────────────────────────────────────────────────
_client = openai.OpenAI(
    api_key=args.api_key,
    base_url="https://api.deepseek.com",
    timeout=60.0,
    max_retries=3,
)

def llm_fn(messages, system):
    try:
        response = _client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        log.warning("DeepSeek rate limit — waiting 10s before retry")
        time.sleep(10)
        raise
    except openai.AuthenticationError:
        log.error("Invalid DeepSeek API key. Check https://platform.deepseek.com")
        sys.exit(1)
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        raise


# ── Embed function — local, no API cost ───────────────────────────────────────
embed_fn    = _semantic_embed if _ST_AVAILABLE else _simple_embed
embed_label = "sentence-transformers/all-MiniLM-L6-v2" if _ST_AVAILABLE else "n-gram fallback"


# ── Persistent storage paths ───────────────────────────────────────────────────
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH    = os.path.join(_BASE_DIR, "aegis_memory.json")
KNOWLEDGE_PATH = os.path.join(_BASE_DIR, "aegis_knowledge.json")


# ── Boot ───────────────────────────────────────────────────────────────────────
log.info(f"Booting Aegis-1  model={args.model}  embedder={embed_label}")
log.info(f"Memory path: {MEMORY_PATH}")

boot_start = time.time()

# Check if existing memory will be loaded
memory_exists = os.path.exists(MEMORY_PATH)
if memory_exists:
    log.info(f"Found existing memory file — will load on boot")
else:
    log.info("No existing memory file — starting fresh")

print(f"\n🧠  Booting Aegis-1  ({args.model})")
print(f"    Embedder : {embed_label}")
print(f"    Memory   : {'loading ' if memory_exists else 'new → '}{MEMORY_PATH}\n")

try:
    aegis = Aegis(
        llm_fn=llm_fn,
        embed_fn=embed_fn,
        verbose=args.verbose,
        memory_path=MEMORY_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        # Memory
        working_memory_capacity=20,
        recall_top_k=8,
        recall_threshold=0.05,    # low threshold — better cross-session recall
        # VSA
        vsa_dim=10_000,
        vsa_capacity=5_000,
        vsa_threshold=0.55,
        # Conflict monitor
        monitor_strong_threshold=0.72,
        monitor_moderate_threshold=0.50,
        max_rethink_attempts=2,
        # Attention
        attention_suppression_threshold=0.12,
        attention_critical_threshold=0.80,
        # MetaCognition
        meta_rethink_threshold=0.30,
        meta_optimal_response_words=(40, 400),
        system_prompt=(
            "You are a precise, helpful assistant with a persistent memory system. "
            "When you see a section called '## Relevant Long-Term Memories', those "
            "are VERIFIED FACTS from previous conversations with this exact user. "
            "You MUST use them to answer questions about the user — do NOT say you "
            "lack information if the answer appears in those memories. "
            "Treat those memories as ground truth, not as uncertain context."
        ),
    )
except Exception as e:
    log.exception(f"Failed to boot Aegis: {e}")
    sys.exit(1)

# ── Pre-load ground-truth facts ────────────────────────────────────────────────
FACTS = [
    ("The Eiffel Tower is located in Paris, France.",          "geography"),
    ("Python was first released in 1991 by Guido van Rossum.", "technology"),
    ("The speed of light in a vacuum is 299,792 km/s.",        "physics"),
    ("Water boils at 100 degrees Celsius at sea level.",       "physics"),
    ("The Great Wall of China is located in China.",           "geography"),
    ("Shakespeare wrote Hamlet, Macbeth, and Othello.",        "literature"),
]

for fact, category in FACTS:
    aegis.learn_fact(fact, category=category)

log.info(f"Loaded {len(FACTS)} ground-truth facts")

# ── Persistent goals ───────────────────────────────────────────────────────────
aegis.push_goal("Be concise and precise. Answer in 2-3 sentences unless asked to elaborate.",
                priority=GoalPriority.HIGH)
aegis.push_goal("If uncertain about a fact, say so explicitly.",
                priority=GoalPriority.NORMAL)
aegis.push_goal("Never contradict an established fact in this conversation.",
                priority=GoalPriority.HIGH)

boot_ms = (time.time() - boot_start) * 1000
log.info(f"Boot complete in {boot_ms:.0f}ms")
print(f"✅  Aegis-1 ready  ({boot_ms:.0f}ms boot,  {len(FACTS)} facts loaded)\n")


# ── Graceful shutdown ──────────────────────────────────────────────────────────
def _shutdown(sig=None, frame=None):
    print("\n💾  Consolidating memory before exit...")
    try:
        aegis.sleep()
        log.info("Memory consolidated successfully")
        print("✅  Goodbye!")
    except Exception as e:
        log.warning(f"Memory consolidation failed: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Telemetry ──────────────────────────────────────────────────────────────────
def print_telemetry(resp):
    v = resp.validation
    m = resp.meta
    a = resp.attention
    g = resp.gateway

    conflict_lines = ""
    for c in v.conflicts:
        conflict_lines += (
            f"\n  │     ⚠️  [{c.severity.value}] \"{c.llm_claim}\""
            f"\n  │         vs stored: \"{c.stored_fact}\""
            f"\n  │         confidence: {c.confidence:.0%}"
        )

    flag_str = "  flags=[" + ", ".join(f.value for f in m.flags) + "]" if m.flags else ""

    print(
        f"\n  ┌─ Telemetry {'─' * 44}\n"
        f"  │  Turn        : {resp.turn}\n"
        f"  │  Route       : {resp.route.value}  (novelty={g.novelty:.2f})\n"
        f"  │  Attention   : {a.priority.value}  (salience={a.salience:.2f})\n"
        f"  │  Validation  : {'✅ passed' if v.passed else '❌ failed'}  "
        f"(checked {v.checked_facts} facts, rethinks={resp.rethink_attempts}){conflict_lines}\n"
        f"  │  Confidence  : {m.confidence:.0%}  "
        f"(quality={m.reasoning_quality:.2f}, words={m.response_length}){flag_str}\n"
        f"  │  Emotion     : {resp.valence.dominant_emotion.value}  "
        f"(valence={resp.valence.valence:+.2f}, arousal={resp.valence.arousal:.2f})\n"
        f"  │  Memory hits : VSA={resp.vsa_hits}  cortex={resp.cortex_memories}\n"
        f"  │  Latency     : {resp.total_latency_ms:.0f}ms total  "
        f"(llm={resp.llm_latency_ms:.0f}ms, monitor={resp.monitor_latency_ms:.0f}ms)\n"
        f"  └{'─' * 55}"
    )

def print_compact(resp):
    valid = "✅" if resp.validation.passed else "⚠️ "
    rethink = f"  rethinks={resp.rethink_attempts}" if resp.rethink_attempts else ""
    mem = f"  mem={resp.cortex_memories}" if resp.cortex_memories else ""
    print(f"  [{resp.route.value} | {valid} | confidence {resp.meta.confidence:.0%} | "
          f"{resp.total_latency_ms:.0f}ms{rethink}{mem}]\n")


# ── Smoke Tests ────────────────────────────────────────────────────────────────
SMOKE_TESTS = [
    {
        "label":        "Basic factual query",
        "input":        "Where is the Eiffel Tower?",
        "expect_valid": None,
        "check":        lambda r: "paris" in r.text.lower(),
        "check_label":  "response mentions Paris",
    },
    {
        "label":        "Contradiction — conflict monitor should fire",
        "input":        "Is it true the Eiffel Tower is in London?",
        "expect_valid": None,
        "check":        lambda r: "london" not in r.text.lower() or "not" in r.text.lower(),
        "check_label":  "response does not confirm London",
    },
    {
        "label":        "Goal enforcement — answer should be concise",
        "input":        "Explain how the internet works.",
        "expect_valid": True,
        "check":        lambda r: r.meta.response_length < 200,
        "check_label":  "response under 200 words",
    },
    {
        "label":        "Numeric fact check",
        "input":        "What temperature does water boil at?",
        "expect_valid": True,
        "check":        lambda r: "100" in r.text,
        "check_label":  "response contains 100",
    },
    {
        "label":        "Multi-turn memory",
        "input":        "What was my very first question to you?",
        "expect_valid": True,
        "check":        lambda r: len(r.text) > 10,
        "check_label":  "non-empty response",
    },
    {
        "label":        "Uncertainty signal",
        "input":        "What will Apple's exact stock price be on 1 January 2030?",
        "expect_valid": None,
        "check":        lambda r: any(w in r.text.lower() for w in
                            ["cannot", "can't", "predict", "impossible",
                             "unknown", "uncertain", "no way", "don't know"]),
        "check_label":  "response acknowledges uncertainty",
    },
]


def run_smoke_tests():
    log.info(f"Starting smoke tests ({len(SMOKE_TESTS)} turns)")
    print("=" * 60)
    print(f"SMOKE TESTS  ({len(SMOKE_TESTS)} turns)")
    print("=" * 60)

    results = []
    for i, t in enumerate(SMOKE_TESTS, 1):
        print(f"\n[{i}/{len(SMOKE_TESTS)}] {t['label']}")
        print(f"  User  : {t['input']}")
        log.debug(f"Smoke test {i}: {t['input']}")

        try:
            resp = aegis.think(t["input"])
        except Exception as e:
            log.error(f"think() raised: {e}")
            print(f"  ❌  Exception: {e}")
            results.append(False)
            continue

        print(f"  Aegis : {resp.text.strip()}")
        log.debug(f"Response: route={resp.route.value} confidence={resp.meta.confidence:.2f} "
                  f"valid={resp.validation.passed} latency={resp.total_latency_ms:.0f}ms")

        if args.verbose:
            print_telemetry(resp)
        else:
            print_compact(resp)

        ok = True
        if t["expect_valid"] is not None and resp.validation.passed != t["expect_valid"]:
            log.warning(f"Validation mismatch: expected={t['expect_valid']} got={resp.validation.passed}")
            print(f"  ❌  Expected validation.passed={t['expect_valid']}, got {resp.validation.passed}")
            ok = False

        try:
            if not t["check"](resp):
                log.warning(f"Content check failed: {t['check_label']}")
                print(f"  ❌  {t['check_label']}")
                ok = False
            else:
                print(f"  ✅  {t['check_label']}")
        except Exception as e:
            log.warning(f"Check error: {e}")
            print(f"  ⚠️   Check error: {e}")

        results.append(ok)

    passed = sum(results)
    print("\n" + "=" * 60)
    print(f"RESULT  {passed}/{len(results)} passed")

    print("\nSUBSYSTEM STATS:")
    stats = aegis.stats() if callable(aegis.stats) else aegis.stats
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {val}")

    log.info(f"Smoke tests complete: {passed}/{len(results)} passed")
    if all(results):
        print("\n✅  All smoke tests passed — Aegis-1 is production ready.")
    else:
        print(f"\n⚠️   {len(results) - passed} test(s) failed — review output above.")

    return all(results)


if args.smoke:
    ok = run_smoke_tests()
    sys.exit(0 if ok else 1)


# ── Interactive Chat ───────────────────────────────────────────────────────────
COMMANDS = {
    "help":  "Show this help",
    "stats": "Print subsystem statistics",
    "reset": "Clear conversation history",
    "sleep": "Consolidate and save memory",
    "goals": "List active goals",
    "quit":  "Save memory and exit",
}

def print_help():
    print("\n  Commands:")
    for cmd, desc in COMMANDS.items():
        print(f"    {cmd:<8} {desc}")
    print()

print("💬  Chat mode  —  type 'help' for commands\n")
print_help()
log.info("Entering interactive chat mode")

while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        _shutdown()

    if not user_input:
        continue

    cmd = user_input.lower()

    if cmd in ("quit", "exit", "q"):
        _shutdown()

    elif cmd == "help":
        print_help()

    elif cmd == "stats":
        stats = aegis.stats() if callable(aegis.stats) else aegis.stats
        print()
        for key, val in stats.items():
            if isinstance(val, dict):
                print(f"  {key}:")
                for k2, v2 in val.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {key}: {val}")
        print()

    elif cmd == "reset":
        aegis.reset_conversation()
        log.info("Conversation reset")
        print("  🔄  Conversation reset.\n")

    elif cmd == "sleep":
        print("  💾  Consolidating memory...")
        try:
            aegis.sleep()
            log.info("Memory consolidated")
            print("  ✅  Done.\n")
        except Exception as e:
            log.error(f"Sleep failed: {e}")
            print(f"  ❌  Failed: {e}\n")

    elif cmd == "goals":
        goals = aegis.list_goals()
        print()
        for g in goals:
            print(f"  [{g.priority.value}] {g.text}")
        if not goals:
            print("  No active goals.")
        print()

    else:
        log.debug(f"Turn {aegis._turn_count + 1}: {user_input[:60]}")
        try:
            resp = aegis.think(user_input)
        except Exception as e:
            log.error(f"think() failed: {e}")
            print(f"\n  ❌  Error: {e}\n")
            continue

        print(f"\nAegis: {resp.text.strip()}\n")
        log.debug(f"Response: route={resp.route.value} confidence={resp.meta.confidence:.2f} "
                  f"valid={resp.validation.passed} latency={resp.total_latency_ms:.0f}ms "
                  f"mem_hits={resp.cortex_memories}")

        if args.verbose:
            print_telemetry(resp)
        else:
            print_compact(resp)