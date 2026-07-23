from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from .llm_json import complete_json
from .models import ResearchIntent

logger = logging.getLogger(__name__)


class ElicitationAgent:
    """Human externalization: surface latent research intent without blocking the API."""

    def __init__(self, client: Any, model: str = "gpt-5.2") -> None:
        self._client = client
        self._model = model

    def elicit(self, question: str, *, strict: bool = False) -> ResearchIntent:
        cleaned = (question or "").strip()
        if not cleaned:
            return ResearchIntent(
                objective="",
                refined_question="",
                is_sufficiently_specified=False,
                clarifying_questions=["What research question are you trying to answer?"],
                discovery_type="relationship",
            )

        system = (
            "You are the Elicitation Agent in a Knowledge Discovery Ecosystem. "
            "Infer a structured ResearchIntent from a scholar's prompt. "
            "Identify vague or underspecified questions and propose targeted clarifying_questions, "
            "but the pipeline will proceed anyway — always provide a best-effort refined_question, "
            "objective, target_concepts, theoretical_contribution, assumptions, and discovery_type. "
            "discovery_type should be one of: definition, relationship, proposition, research_gap, "
            "conceptual_model, or a short custom label. Return JSON only. "
            "Keep compact: target_concepts <= 6; clarifying_questions <= 4; assumptions <= 4; "
            "objective/refined_question/theoretical_contribution each <= 300 chars."
        )
        user = json.dumps({"user_prompt": cleaned}, ensure_ascii=False, indent=2)
        try:
            intent = complete_json(
                self._client,
                model=self._model,
                system=system,
                user=user,
                schema_model=ResearchIntent,
            )
            if not (intent.refined_question or "").strip():
                intent = intent.model_copy(update={"refined_question": cleaned})
            return intent
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug("Elicitation LLM returned invalid output: %s", exc)
            if strict:
                raise RuntimeError(f"Elicitation primary LLM path failed (strict): {exc}") from exc
        except Exception as exc:
            logger.warning("Elicitation LLM failed: %s: %s", type(exc).__name__, exc)
            if strict:
                raise RuntimeError(f"Elicitation primary LLM path failed (strict): {exc}") from exc

        return ResearchIntent(
            objective=f"Investigate: {cleaned}",
            target_concepts=[],
            theoretical_contribution="",
            assumptions=[],
            discovery_type="relationship",
            clarifying_questions=[],
            is_sufficiently_specified=True,
            refined_question=cleaned,
        )
