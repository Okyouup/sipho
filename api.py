"""
Aegis-1 — FastAPI Web API
==========================
Wraps aegis.think() as an HTTP endpoint.

Start:
    uvicorn api:app --host 0.0.0.0 --port 8000

Env vars required:
    DEEPSEEK_API_KEY   your DeepSeek key
    OBS_ACCESS_KEY     Huawei OBS access key      (optional)
    OBS_SECRET_KEY     Huawei OBS secret key      (optional)
    OBS_BUCKET         Huawei OBS bucket name     (optional)
    OBS_ENDPOINT       e.g. obs.af-south-1.myhuaweicloud.com (optional)
"""

import os
import sys
import time
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
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
        }
    },
    "loggers": {
        "aegis.api":               {"level": "DEBUG", "handlers": ["console"], "propagate": False},
        "openai":                  {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "httpx":                   {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "httpcore":                {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "huggingface_hub":         {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "sentence_transformers":   {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "transformers":            {"level": "ERROR", "handlers": ["console"], "propagate": False},
        "nengo":                   {"level": "CRITICAL","handlers": ["console"], "propagate": False},
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
})

log = logging.getLogger("aegis.api")

# ── Path ───────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Aegis imports ──────────────────────────────────────────────────────────────
import openai
from Aegis import Aegis
from Goalstack import GoalPriority
from Llm import _semantic_embed, _simple_embed, _ST_AVAILABLE

# ── OBS sync (optional) ────────────────────────────────────────────────────────
try:
    from obs_sync import OBSMemorySync
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("DEEPSEEK_API_KEY", "")
MODEL        = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH  = os.path.join(BASE_DIR, "aegis_memory.json")
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "aegis_knowledge.json")

FACTS = [
    ("The Eiffel Tower is located in Paris, France.",          "geography"),
    ("Python was first released in 1991 by Guido van Rossum.", "technology"),
    ("The speed of light in a vacuum is 299,792 km/s.",        "physics"),
    ("Water boils at 100 degrees Celsius at sea level.",       "physics"),
]

# ── Global Aegis instance ──────────────────────────────────────────────────────
aegis: Optional[Aegis] = None
obs_sync: Optional[object] = None
boot_time: float = 0.0


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global aegis, obs_sync, boot_time
    log.info("Booting Aegis-1...")
    t0 = time.time()

    # Pull memory from OBS if available
    if OBS_AVAILABLE and all([
        os.getenv("OBS_ACCESS_KEY"),
        os.getenv("OBS_SECRET_KEY"),
        os.getenv("OBS_BUCKET"),
        os.getenv("OBS_ENDPOINT"),
    ]):
        obs_sync = OBSMemorySync(
            access_key=os.getenv("OBS_ACCESS_KEY"),
            secret_key=os.getenv("OBS_SECRET_KEY"),
            bucket=os.getenv("OBS_BUCKET"),
            endpoint=os.getenv("OBS_ENDPOINT"),
            local_memory_path=MEMORY_PATH,
            local_knowledge_path=KNOWLEDGE_PATH,
        )
        obs_sync.pull()
        log.info("OBS memory pulled to local disk")
    else:
        log.info("OBS not configured — using local memory only")

    # Build LLM function
    if not API_KEY:
        log.error("DEEPSEEK_API_KEY not set")
        raise RuntimeError("DEEPSEEK_API_KEY environment variable required")

    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=3,
    )

    def llm_fn(messages, system):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    embed_fn = _semantic_embed if _ST_AVAILABLE else _simple_embed

    aegis = Aegis(
        llm_fn=llm_fn,
        embed_fn=embed_fn,
        memory_path=MEMORY_PATH,
        knowledge_path=KNOWLEDGE_PATH,
        working_memory_capacity=20,
        recall_top_k=8,
        recall_threshold=0.05,
        vsa_dim=10_000,
        vsa_capacity=5_000,
        vsa_threshold=0.55,
        monitor_strong_threshold=0.78,
        monitor_moderate_threshold=0.66,
        max_rethink_attempts=2,
        system_prompt=(
            "You are a precise, helpful assistant with a persistent memory system. "
            "When you see '## Relevant Long-Term Memories', those are VERIFIED FACTS "
            "from previous conversations — treat them as ground truth and use them to "
            "answer questions. Never say you lack information if it appears in memories."
        ),
    )

    for fact, category in FACTS:
        aegis.learn_fact(fact, category=category)

    aegis.push_goal("Be concise. Answer in 2-3 sentences unless asked to elaborate.",
                    priority=GoalPriority.HIGH)
    aegis.push_goal("If uncertain, say so explicitly.", priority=GoalPriority.NORMAL)

    boot_time = (time.time() - t0) * 1000
    log.info(f"Aegis-1 ready in {boot_time:.0f}ms")

    yield  # ── app is running ──

    # Shutdown — consolidate memory then push to OBS
    log.info("Shutting down — consolidating memory...")
    try:
        aegis.sleep()
        log.info("Memory consolidated")
    except Exception as e:
        log.warning(f"sleep() failed: {e}")

    if obs_sync:
        obs_sync.push()
        log.info("Memory pushed to OBS")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aegis-1 Cognitive API",
    description="Neuromorphic cognitive middleware for LLMs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    text: str
    route: str
    confidence: float
    validation_passed: bool
    rethink_attempts: int
    memory_hits: int
    latency_ms: float
    conflicts: list
    novelty: float = 0.0
    vsa_hits: int = 0
    attention_suppressed: bool = False

class FactRequest(BaseModel):
    fact: str
    category: str = "general"

class GoalRequest(BaseModel):
    text: str
    priority: str = "normal"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    ui_path = os.path.join(BASE_DIR, "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Aegis-1 API running</h1><p>See /docs for API reference.</p>")


@app.get("/health")
async def health():
    """Health check — used by Huawei ECS load balancer."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    stats = aegis.stats() if callable(aegis.stats) else aegis.stats
    return {
        "status": "ok",
        "boot_ms": boot_time,
        "turns": stats.get("turns", 0),
        "memory_facts": stats.get("monitor", {}).get("knowledge_facts", 0),
        "obs_connected": obs_sync is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")

    log.debug(f"[{req.session_id}] {req.message[:60]}")

    try:
        resp = aegis.think(req.message)
    except Exception as e:
        log.error(f"think() failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    conflicts = [
        {
            "claim": c.llm_claim,
            "stored_fact": c.stored_fact,
            "severity": c.severity.value,
            "confidence": round(c.confidence, 3),
        }
        for c in resp.validation.conflicts
    ]

    log.debug(
        f"route={resp.route.value} confidence={resp.meta.confidence:.2f} "
        f"valid={resp.validation.passed} latency={resp.total_latency_ms:.0f}ms"
    )

    return ChatResponse(
        text=resp.text,
        route=resp.route.value,
        confidence=round(resp.meta.confidence, 3),
        validation_passed=resp.validation.passed,
        rethink_attempts=resp.rethink_attempts,
        memory_hits=resp.cortex_memories,
        latency_ms=round(resp.total_latency_ms, 1),
        conflicts=conflicts,
        novelty=round(resp.gateway.novelty, 3),
        vsa_hits=resp.vsa_hits,
        attention_suppressed=resp.attention.suppressed,
    )


@app.post("/facts")
async def add_fact(req: FactRequest):
    """Load a ground-truth fact into the conflict monitor."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    aegis.learn_fact(req.fact, category=req.category)
    log.info(f"Fact loaded: [{req.category}] {req.fact}")
    return {"status": "ok", "fact": req.fact, "category": req.category}


@app.post("/goals")
async def add_goal(req: GoalRequest):
    """Push a persistent goal."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    priority_map = {
        "low":      GoalPriority.LOW,
        "normal":   GoalPriority.NORMAL,
        "high":     GoalPriority.HIGH,
    }
    priority = priority_map.get(req.priority.lower(), GoalPriority.NORMAL)
    goal = aegis.push_goal(req.text, priority=priority)
    log.info(f"Goal pushed: [{req.priority}] {req.text}")
    return {"status": "ok", "goal_id": goal.goal_id, "text": goal.text}


@app.post("/sleep")
async def sleep():
    """Consolidate memory and push to OBS."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    try:
        aegis.sleep()
        log.info("Memory consolidated via /sleep")
        if obs_sync:
            obs_sync.push()
            log.info("Memory pushed to OBS via /sleep")
        return {"status": "ok", "obs_synced": obs_sync is not None}
    except Exception as e:
        log.error(f"sleep() failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    """Reset conversation history (keeps long-term memory)."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    aegis.reset_conversation()
    log.info("Conversation reset via /reset")
    return {"status": "ok"}


@app.get("/goals")
async def list_goals():
    """Return all active goals."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    return {
        "goals": [
            {
                "goal_id": g.goal_id,
                "text": g.text,
                "priority": g.priority.name,
            }
            for g in aegis.list_goals()
        ]
    }


@app.delete("/goals/{goal_id}")
async def delete_goal(goal_id: str):
    """Remove a goal by ID."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    removed = aegis.remove_goal(goal_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Goal not found")
    log.info(f"Goal removed: {goal_id}")
    return {"status": "ok", "removed": goal_id}


@app.get("/stats")
async def stats():
    """Return full subsystem stats."""
    if aegis is None:
        raise HTTPException(status_code=503, detail="Aegis not initialised")
    return aegis.stats() if callable(aegis.stats) else aegis.stats