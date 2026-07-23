from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .llm_json import complete_json
from .models import IntegrationState, ReflectionDecision, ResearchIntent

logger = logging.getLogger(__name__)


class ReflectionAgent:
    """Agent-side iterative refinement decision (paper collaboration cycle 6/7)."""

    def __init__(self, client: Any, model: str = "gpt-5.2") -> None:
        self._client = client
        self._model = model

    def reflect(
        self,
        *,
        question: str,
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        integration_state: Optional[Dict[str, Any] | IntegrationState],
        analysis_brief: str,
        already_run_analyses: List[str],
        allowed_analyses: List[str],
        cycle_index: int,
        max_cycles: int,
        kg_excerpt: str = "",
        chunk_excerpt: str = "",
        strict: bool = False,
    ) -> ReflectionDecision:
        intent_payload: Dict[str, Any]
        if isinstance(research_intent, ResearchIntent):
            intent_payload = research_intent.model_dump()
        elif isinstance(research_intent, dict):
            intent_payload = research_intent
        else:
            intent_payload = {}

        integ_payload: Dict[str, Any]
        if isinstance(integration_state, IntegrationState):
            integ_payload = integration_state.model_dump()
        elif isinstance(integration_state, dict):
            integ_payload = integration_state
        else:
            integ_payload = {}

        exhausted = (cycle_index + 1) >= max_cycles
        system = (
            "You are the Reflection Agent. Evaluate whether another retrieval/analysis cycle "
            "is worthwhile for answering the research intent. Judging criteria: evidence sufficiency, "
            "remaining conceptual uncertainty, and which Analysaurus analyses would most improve knowledge. "
            "Set continue_loop=true only if another cycle is likely to add material new knowledge AND "
            "cycles remain. Set sufficient=true when a high-quality knowledge package can already be built. "
            "recommended_analyses must use function_name values from allowed_analyses and should avoid "
            "already_run_analyses unless strongly justified. Return JSON only."
        )
        user = json.dumps(
            {
                "user_question": question,
                "research_intent": intent_payload,
                "integration_state": integ_payload,
                "analysis_brief": analysis_brief,
                "already_run_analyses": already_run_analyses,
                "allowed_analyses": allowed_analyses,
                "cycle_index": cycle_index,
                "max_cycles": max_cycles,
                "cycles_remaining_after_this": max(0, max_cycles - cycle_index - 1),
                "kg_excerpt": kg_excerpt,
                "chunk_excerpt": chunk_excerpt,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            decision = complete_json(
                self._client,
                model=self._model,
                system=system,
                user=user,
                schema_model=ReflectionDecision,
            )
            allowed = set(allowed_analyses)
            completed = set(already_run_analyses)
            recs = [a for a in decision.recommended_analyses if a in allowed and a not in completed]
            if not recs:
                recs = [a for a in decision.recommended_analyses if a in allowed][:3]
            continue_loop = bool(decision.continue_loop) and not exhausted and not bool(decision.sufficient)
            if exhausted:
                continue_loop = False
            return decision.model_copy(
                update={
                    "recommended_analyses": recs,
                    "continue_loop": continue_loop,
                    "sufficient": bool(decision.sufficient) or exhausted,
                }
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug("Reflection LLM returned invalid output: %s", exc)
            if strict:
                raise RuntimeError(f"Reflection primary LLM path failed (strict): {exc}") from exc
        except Exception as exc:
            logger.warning("Reflection LLM failed: %s: %s", type(exc).__name__, exc)
            if strict:
                raise RuntimeError(f"Reflection primary LLM path failed (strict): {exc}") from exc

        confidence = float(integ_payload.get("confidence") or 0.0)
        enough = confidence >= 0.7 or exhausted
        return ReflectionDecision(
            sufficient=enough,
            continue_loop=not enough and not exhausted,
            uncertainties=["Reflection LLM unavailable; using heuristic stop rule."],
            recommended_analyses=[],
            follow_up_question="",
            rationale="Fallback reflection after LLM failure.",
        )
