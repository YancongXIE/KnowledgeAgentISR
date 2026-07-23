from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


DEFAULT_TTL_SECONDS = 3600


@dataclass
class PausedSession:
    session_id: str
    phase: str  # elicitation | human_collaboration
    graph_state: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    research_intent: Optional[Dict[str, Any]] = None
    clarifying_questions: list[str] = field(default_factory=list)
    collaboration_prompt: str = ""
    interactive: bool = True
    use_kg: bool = True
    compare_baseline: bool = False


class SessionStore:
    """In-process pause/resume store for interactive KDE runs."""

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = max(60, int(ttl_seconds))
        self._lock = threading.Lock()
        self._sessions: Dict[str, PausedSession] = {}

    def create(
        self,
        *,
        phase: str,
        graph_state: Dict[str, Any],
        research_intent: Optional[Dict[str, Any]] = None,
        clarifying_questions: Optional[list[str]] = None,
        collaboration_prompt: str = "",
        interactive: bool = True,
        use_kg: bool = True,
        compare_baseline: bool = False,
        session_id: Optional[str] = None,
    ) -> PausedSession:
        sid = (session_id or "").strip() or str(uuid.uuid4())
        session = PausedSession(
            session_id=sid,
            phase=phase,
            graph_state=dict(graph_state),
            research_intent=dict(research_intent) if isinstance(research_intent, dict) else None,
            clarifying_questions=list(clarifying_questions or []),
            collaboration_prompt=collaboration_prompt or "",
            interactive=bool(interactive),
            use_kg=bool(use_kg),
            compare_baseline=bool(compare_baseline),
        )
        with self._lock:
            self._purge_locked()
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[PausedSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            self._purge_locked()
            return self._sessions.get(sid)

    def pop(self, session_id: str) -> Optional[PausedSession]:
        sid = (session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            self._purge_locked()
            return self._sessions.pop(sid, None)

    def update(self, session: PausedSession) -> None:
        with self._lock:
            self._purge_locked()
            session.created_at = time.time()
            self._sessions[session.session_id] = session

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.created_at > self._ttl]
        for sid in expired:
            self._sessions.pop(sid, None)


# Process-wide store used by the web API / runtime.
GLOBAL_SESSION_STORE = SessionStore()
