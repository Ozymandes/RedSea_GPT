"""RedSea GPT web API — FastAPI backend for the interactive demo.

Single-host design: this process serves BOTH the JSON API (/api/*) and the
built React frontend (static files at /). One process, one deploy.

The API wraps the existing generation.RedSeaGPT (baseline) and
generation.RedSeaAgent (CRAG) under one uniform contract, adds per-session
conversation memory, and surfaces the reasoning trace / retrieved chunks so the
frontend can show *how* the system reasoned — not just the answer.

Security:
- The OptiLLM key is read from env on the server and NEVER sent to the client.
- The corpus PDFs are NOT served (copyright). Only extracted chunk quotes with
  source + page are returned, which is exactly what the citations need.
- CORS is locked to same-origin in production (the static frontend is served by
  this same process). A permissive dev override exists for local Vite hot-reload.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("redsea.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# --- engine + memory (lazy import so the module loads even if heavy deps lag) ---
from generation import RedSeaGPT, RedSeaAgent  # noqa: E402
from generation.memory import ConversationMemory, Turn  # noqa: E402

ENGINE_MODE = os.getenv("REDSEA_ENGINE", "baseline").strip().lower()  # "baseline" | "agent"
MAX_SESSIONS = int(os.getenv("REDSEA_MAX_SESSIONS", "200"))

# ---------------------------------------------------------------------------
# Session store (in-memory; fine for a demo. A real deploy would use Redis.)
# ---------------------------------------------------------------------------
class SessionStore:
    """Bounded in-memory store of conversation memories keyed by session id."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._mems: Dict[str, ConversationMemory] = {}
        self._max = max_sessions

    def get_or_create(self, session_id: Optional[str]) -> ConversationMemory:
        if session_id and session_id in self._mems:
            return self._mems[session_id]
        sid = session_id or uuid.uuid4().hex[:12]
        mem = ConversationMemory(max_turns=6)
        self._mems[sid] = mem
        # Evict oldest if we grew past the cap (simple FIFO; fine for demo).
        if len(self._mems) > self._max:
            oldest = next(iter(self._mems))
            if oldest != sid:
                self._mems.pop(oldest, None)
        return mem

    def reset(self, session_id: str) -> None:
        self._mems.pop(session_id, None)


SESSIONS = SessionStore()


# ---------------------------------------------------------------------------
# Engine factory (built once at startup; loading the vectorstore is the slow bit)
# ---------------------------------------------------------------------------
engine: Any = None


def build_engine() -> Any:
    mode = ENGINE_MODE
    kwargs: Dict[str, Any] = {}
    if mode == "agent":
        logger.info("Building CRAG agent engine (LangGraph).")
        return RedSeaAgent(**kwargs)
    logger.info("Building baseline RAG engine.")
    return RedSeaGPT(retrieval_k=int(os.getenv("REDSEA_K", "7")), **kwargs)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    agent: Optional[bool] = None  # per-request override of REDSEA_ENGINE


class Citation(BaseModel):
    citation_id: int
    source: str
    page: Any
    content: str
    # quote is the short, front-end-facing excerpt (already truncated upstream)


class ReasoningTrace(BaseModel):
    """The 'how' behind an answer. All fields optional — baseline path populates
    a subset, the CRAG agent populates the full graph trace."""
    route: Optional[str] = None
    retrieval_method: Optional[str] = None
    retrieval_rounds: Optional[int] = None
    confidence: Optional[float] = None
    resolved_question: Optional[str] = None  # shows pronoun resolution (memory wow)
    verification: Optional[Dict[str, Any]] = None
    node_trace: Optional[Dict[str, float]] = None
    reason: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    citations: List[Citation]
    reasoning: ReasoningTrace
    session_id: str
    engine: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="RedSea GPT", version="1.0.0", docs_url="/api/docs")

# Dev CORS: allow the Vite dev server (5173) to call the API during local dev.
# In production the frontend is served by this same process (same origin).
_dev_origins = os.getenv("REDSEA_DEV_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _dev_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global engine
    if engine is None:
        engine = build_engine()


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "engine": ENGINE_MODE,
        "engine_loaded": engine is not None,
        "active_sessions": len(SESSIONS._mems),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if engine is None:  # safety; startup should have built it
        raise HTTPException(503, "Engine not ready yet")

    # Decide engine per-request only if we built the agent; baseline is default.
    # (Swapping engines per-request would mean loading two vectorstores; we keep
    # it simple: the process runs ONE engine, set by REDSEA_ENGINE.)
    use_agent = ENGINE_MODE == "agent"

    mem = SESSIONS.get_or_create(req.session_id)
    try:
        result = engine.query(req.message, return_source_docs=True, memory=mem)
    except Exception as exc:  # noqa: BLE001 - never crash the API
        logger.exception("Engine query failed")
        return ChatResponse(
            answer="",
            refused=False,
            citations=[],
            reasoning=ReasoningTrace(),
            session_id=_sid_from_mem(mem),
            engine=ENGINE_MODE,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    # Record the turn for multiturn memory (so follow-ups can resolve pronouns).
    mem.add(
        Turn(
            question=req.message,
            answer=str(result.get("answer", "")),
            sources=result.get("sources", []) or [],
            resolved_question=result.get("resolved_question"),
        )
    )

    citations = [
        Citation(
            citation_id=int(c.get("citation_id", i + 1)),
            source=str(c.get("source", "Unknown")),
            page=c.get("page"),
            content=str(c.get("content", c.get("page_content", "")))[:400],
        )
        for i, c in enumerate(result.get("sources", []) or [])
    ]

    reasoning = ReasoningTrace(
        route=result.get("route"),
        retrieval_method=result.get("retrieval_method"),
        retrieval_rounds=result.get("retrieval_rounds"),
        confidence=_round(result.get("confidence"), 3),
        resolved_question=result.get("resolved_question"),
        verification=result.get("verification") or None,
        node_trace=result.get("trace") or None,
        reason=result.get("reason"),
    )

    return ChatResponse(
        answer=str(result.get("answer", "")),
        refused=bool(result.get("refusal", False)),
        citations=citations,
        reasoning=reasoning,
        session_id=_sid_from_mem(mem),
        engine=("agent" if use_agent else "baseline"),
    )


@app.post("/api/reset")
def reset(req: ChatRequest) -> Dict[str, Any]:
    if req.session_id:
        SESSIONS.reset(req.session_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static frontend (built React app). Served last so /api/* takes precedence.
# ---------------------------------------------------------------------------
_WEB_DIR = os.getenv("REDSEA_WEB_DIR", os.path.join(os.path.dirname(__file__), "..", "web", "dist"))
_WEB_DIR = os.path.abspath(_WEB_DIR)


@app.get("/")
def index() -> Any:
    idx = os.path.join(_WEB_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return JSONResponse(
        {"message": "RedSea GPT API is running. Frontend not built yet.", "docs": "/api/docs"},
        status_code=200,
    )


# Mount static assets if the frontend has been built.
if os.path.isdir(_WEB_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(_WEB_DIR, "assets")), name="assets")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _round(v: Any, n: int) -> Optional[float]:
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _sid_from_mem(mem: ConversationMemory) -> str:
    """Recover the session id for the given memory object (reverse lookup)."""
    for sid, m in SESSIONS._mems.items():
        if m is mem:
            return sid
    return ""
