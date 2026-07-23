from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver
from pydantic import ValidationError

from .extraction_agent import ExtractionAgent
from .llm_json import complete_json
from .models import IntegrationState, ResearchIntent, VicariousLearningOutput, VicariousReadingItem

logger = logging.getLogger(__name__)


def _item_from_row(row: Dict[str, Any], why: str) -> VicariousReadingItem:
    text = ""
    for key in ("text", "chunk_text", "passage"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            text = val.strip()
            break
    doi = row.get("doi") or row.get("DOI") or ""
    if not isinstance(doi, str):
        doi = str(doi) if doi else ""
    label = doi or (row.get("chunk_uuid") and f"chunk:{row.get('chunk_uuid')}") or "retrieved passage"
    return VicariousReadingItem(
        title_or_label=str(label),
        doi=doi,
        why_useful=why,
        excerpt=text[:900],
    )


class VicariousLearningAgent:
    """Human internalization support: retrieve qualitative / contextual readings."""

    def __init__(self, client: Any, extraction: ExtractionAgent, model: str = "gpt-5.2") -> None:
        self._client = client
        self._extraction = extraction
        self._model = model

    def _build_instruction(
        self,
        *,
        question: str,
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        integration_state: Optional[Dict[str, Any] | IntegrationState],
    ) -> str:
        if isinstance(research_intent, ResearchIntent):
            intent = research_intent.model_dump()
        elif isinstance(research_intent, dict):
            intent = research_intent
        else:
            intent = {}
        if isinstance(integration_state, IntegrationState):
            integ = integration_state.model_dump()
        elif isinstance(integration_state, dict):
            integ = integration_state
        else:
            integ = {}

        concepts = integ.get("emerging_concepts") or intent.get("target_concepts") or []
        props = integ.get("propositions") or []
        gaps = integ.get("theoretical_gaps") or []
        concept_bit = ", ".join(str(c) for c in concepts[:8]) if concepts else question
        return (
            f"Find qualitative, narrative-rich, or case-based passages that help a researcher "
            f"internalize synthesized knowledge about: {concept_bit}. "
            f"Prefer interviews, ethnography, case studies, field observations, and rich contextual "
            f"descriptions. Also surface contradictory qualitative evidence when present. "
            f"Related propositions: {'; '.join(str(p) for p in props[:4])}. "
            f"Gaps: {'; '.join(str(g) for g in gaps[:4])}."
        )

    def learn(
        self,
        driver: Driver,
        *,
        question: str,
        question_embedding: Optional[List[float]],
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        integration_state: Optional[Dict[str, Any] | IntegrationState],
        fallback_records: Optional[List[Dict[str, Any]]] = None,
        strict: bool = False,
        passages: Optional[List[Dict[str, Any]]] = None,
    ) -> VicariousLearningOutput:
        if passages is not None:
            rows = list(passages)
            note = "caller_supplied_passages"
        else:
            instruction = self._build_instruction(
                question=question,
                research_intent=research_intent,
                integration_state=integration_state,
            )
            rows, note = self._extraction.extract_from_instruction(
                driver,
                instruction,
                question=question,
                question_embedding=question_embedding,
                fallback_records=fallback_records,
            )

        system = (
            "You are the Vicarious Learning Agent. Organize a short, practical reading list that helps "
            "a researcher internalize the synthesized concepts through qualitative/contextual immersion. "
            "Populate illustrative_studies, case_studies, narratives, practical_examples, "
            "contradictory_evidence, and a short reading_sequence. Use only the provided passages; "
            "do not invent DOIs or studies. Return JSON only. "
            "Write for a busy scholar: why_useful must be one concrete sentence (what insight this gives). "
            "Keep compact: at most 2 items per list; excerpts <= 140 chars; why_useful <= 100 chars; "
            "reading_sequence <= 4 ordered sources."
        )
        passage_payload = []
        for i, row in enumerate(rows[:6]):
            passage_payload.append(
                {
                    "index": i + 1,
                    "doi": row.get("doi") or row.get("DOI"),
                    "score": row.get("score"),
                    "text": (row.get("text") or "")[:500],
                }
            )
        integ = (
            integration_state.model_dump()
            if isinstance(integration_state, IntegrationState)
            else (integration_state or {})
        )
        user = json.dumps(
            {
                "user_question": question,
                "retrieval_note": note,
                "integration_snapshot": {
                    "emerging_concepts": list(integ.get("emerging_concepts", []) or [])[:6],
                    "propositions": list(integ.get("propositions", []) or [])[:4],
                    "theoretical_gaps": list(integ.get("theoretical_gaps", []) or [])[:4],
                    "merged_context": (integ.get("merged_context") or "")[:800],
                },
                "passages": passage_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        try:
            out = complete_json(
                self._client,
                model=self._model,
                system=system,
                user=user,
                schema_model=VicariousLearningOutput,
            )
            if not out.note:
                out = out.model_copy(update={"note": f"llm_primary|{note}"})
            elif "llm_primary" not in out.note:
                out = out.model_copy(update={"note": f"llm_primary|{out.note}"})
            return out
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug("Vicarious learning LLM returned invalid output: %s", exc)
            if strict:
                raise RuntimeError(f"VicariousLearning primary LLM path failed (strict): {exc}") from exc
        except Exception as exc:
            logger.warning("Vicarious learning LLM failed: %s: %s", type(exc).__name__, exc)
            if strict:
                raise RuntimeError(f"VicariousLearning primary LLM path failed (strict): {exc}") from exc

        items = [_item_from_row(r, "Retrieved qualitative/contextual passage") for r in rows[:8]]
        sequence = [it.doi or it.title_or_label for it in items if (it.doi or it.title_or_label)]
        mid = max(1, len(items) // 3)
        return VicariousLearningOutput(
            illustrative_studies=items[:mid],
            case_studies=items[mid : mid * 2],
            narratives=items[mid * 2 :],
            practical_examples=[],
            contradictory_evidence=[],
            reading_sequence=sequence,
            note=note or "Fallback vicarious packaging from retrieved chunks.",
        )
