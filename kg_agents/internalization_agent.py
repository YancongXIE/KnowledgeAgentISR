from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from neo4j import Driver

from .models import KnowledgePackage, ResearchIntent

logger = logging.getLogger(__name__)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _package_text(package: KnowledgePackage | Dict[str, Any]) -> str:
    if isinstance(package, KnowledgePackage):
        pkg = package
    else:
        try:
            pkg = KnowledgePackage.model_validate(package)
        except Exception:
            return str(package)[:2000]
    concepts = ", ".join(pkg.key_concepts[:12])
    props = "; ".join(pkg.candidate_propositions[:6])
    gaps = "; ".join(pkg.research_gaps[:4])
    return (
        f"{pkg.executive_synthesis}\n"
        f"Concepts: {concepts}\n"
        f"Propositions: {props}\n"
        f"Gaps: {gaps}"
    ).strip()


class InternalizationAgent:
    """⑨ Agent internalization via RAG write-back (not weight distillation)."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    def persist(
        self,
        *,
        package: KnowledgePackage | Dict[str, Any],
        research_intent: Optional[ResearchIntent | Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        text = _package_text(package)
        if not text:
            return {"ok": False, "error": "empty_package_text"}

        concepts: List[str] = []
        synthesis = ""
        if isinstance(package, KnowledgePackage):
            concepts = list(package.key_concepts or [])[:20]
            synthesis = package.executive_synthesis or ""
        elif isinstance(package, dict):
            concepts = [str(c) for c in (package.get("key_concepts") or [])[:20]]
            synthesis = str(package.get("executive_synthesis") or "")[:2000]

        intent_obj = ""
        if isinstance(research_intent, ResearchIntent):
            intent_obj = research_intent.objective or research_intent.refined_question or ""
        elif isinstance(research_intent, dict):
            intent_obj = str(research_intent.get("objective") or research_intent.get("refined_question") or "")

        memory_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        cypher = """
        CREATE (m:AgentMemory {
          uuid: $uuid,
          text: $text,
          executive_synthesis: $synthesis,
          concepts: $concepts,
          question: $question,
          intent_objective: $intent_obj,
          created_at: $created_at,
          source: 'kde_co_construction',
          embedding: $embedding
        })
        RETURN m.uuid AS uuid
        """
        try:
            with self._driver.session() as session:
                rec = session.run(
                    cypher,
                    uuid=memory_id,
                    text=text[:8000],
                    synthesis=synthesis[:2000],
                    concepts=concepts,
                    question=(question or "")[:500],
                    intent_obj=intent_obj[:500],
                    created_at=created_at,
                    embedding=list(embedding) if embedding else None,
                ).single()
            return {"ok": True, "uuid": rec["uuid"] if rec else memory_id}
        except Exception as exc:
            logger.warning("AgentMemory persist failed: %s: %s", type(exc).__name__, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def retrieve(
        self,
        *,
        question_embedding: Optional[List[float]],
        top_k: int = 3,
        query_text: str = "",
    ) -> List[Dict[str, Any]]:
        top_k = max(1, min(int(top_k), 8))
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (m:AgentMemory)
                    RETURN m.uuid AS uuid,
                           m.text AS text,
                           m.executive_synthesis AS executive_synthesis,
                           m.concepts AS concepts,
                           m.question AS question,
                           m.created_at AS created_at,
                           m.embedding AS embedding
                    ORDER BY m.created_at DESC
                    LIMIT $limit
                    """,
                    limit=max(top_k * 8, 24),
                )
                rows = []
                for i, row in enumerate(result):
                    if i >= max(top_k * 8, 24):
                        break
                    if not hasattr(row, "get") and not isinstance(row, dict):
                        break
                    rows.append(row)
        except Exception as exc:
            logger.warning("AgentMemory retrieve failed: %s: %s", type(exc).__name__, exc)
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        q_emb = list(question_embedding) if question_embedding else []
        q_lower = (query_text or "").lower()
        for row in rows:
            emb = row.get("embedding")
            score = 0.0
            if q_emb and isinstance(emb, list) and emb:
                score = _cosine(q_emb, emb)
            elif q_lower:
                hay = f"{row.get('text') or ''} {row.get('question') or ''}".lower()
                score = 0.35 if any(tok and tok in hay for tok in q_lower.split()[:8]) else 0.05
            else:
                score = 0.1
            scored.append(
                (
                    score,
                    {
                        "uuid": row.get("uuid"),
                        "text": (row.get("text") or "")[:1200],
                        "executive_synthesis": (row.get("executive_synthesis") or "")[:600],
                        "concepts": list(row.get("concepts") or [])[:12],
                        "question": row.get("question") or "",
                        "created_at": row.get("created_at") or "",
                        "score": round(score, 4),
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    @staticmethod
    def format_for_prompt(memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        parts = ["Prior co-constructed AgentMemory (internalized knowledge):"]
        for i, mem in enumerate(memories, 1):
            syn = mem.get("executive_synthesis") or mem.get("text") or ""
            parts.append(f"[{i}] score={mem.get('score', 0)} {syn[:400]}")
        return "\n".join(parts)
