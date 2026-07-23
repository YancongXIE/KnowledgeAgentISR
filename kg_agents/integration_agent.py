from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .llm_json import complete_json
from .models import (
    ClaimProvenance,
    ConceptualRelationship,
    IntegrationOutput,
    IntegrationState,
    KnowledgePackage,
    ResearchIntent,
    VicariousLearningOutput,
)

logger = logging.getLogger(__name__)


def _safe_json(value: Any, *, max_chars: int = 12000) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)[:max_chars]
    except Exception:
        return str(value)[:max_chars]


def _window_text(text: str, *, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    marker = "\n\n... *[truncated]* ...\n\n"
    head = max(20, int(max_chars * 0.45))
    tail = max(20, max_chars - head - len(marker))
    return f"{value[:head]}{marker}{value[-tail:]}"


def _kg_rows_markdown(rows: List[Dict[str, Any]], *, max_rows: int) -> str:
    if not rows:
        return "*No graph rows.*"
    lines: List[str] = []
    for i, row in enumerate(rows[:max_rows], start=1):
        try:
            snippet = json.dumps(row, ensure_ascii=False)[:800]
        except (TypeError, ValueError):
            snippet = str(row)[:800]
        lines.append(f"{i}. `{snippet}`")
    return "\n".join(lines)


def _build_integrate_user_markdown(
    *,
    stage: str,
    question: str,
    pdf_summary: str,
    integration_memo: str,
    kg_records: List[Dict[str, Any]],
    analysis_report: Dict[str, Any] | None,
    analysis_note: str,
    max_kg_rows: int,
) -> str:
    excerpt = _safe_json(analysis_report, max_chars=10000)
    return "\n".join(
        [
            "## User question",
            (question or "(not provided)").strip(),
            "",
            "## Pipeline stage",
            stage,
            "",
            "## PDF / chunk evidence summary",
            _window_text(pdf_summary, max_chars=5000),
            "",
            "## Integration memo (running context)",
            _window_text(integration_memo, max_chars=5000),
            "",
            "## Knowledge graph sample rows",
            _kg_rows_markdown(kg_records, max_rows=max_kg_rows),
            "",
            "## Analysis note",
            _window_text(analysis_note, max_chars=1000),
            "",
            "## Analysis report excerpt",
            "Use only as factual support; do not invent beyond it.",
            "",
            "```json",
            excerpt,
            "```",
            "",
            "## Your task",
            "Return a single JSON object matching the required schema. `merged_context` should integrate the above faithfully and stay centered on the user question.",
        ]
    )


def _build_compose_final_user_markdown(
    *,
    question: str,
    integration_memo: str,
    pdf_summary: str,
    kg_records: List[Dict[str, Any]],
    analysis_note: str,
    analysis_report: Dict[str, Any] | None,
    max_kg_rows: int,
) -> str:
    excerpt = _safe_json(analysis_report, max_chars=9000)
    return "\n".join(
        [
            "## User question (answer this directly)",
            (question or "(not provided)").strip(),
            "",
            "## Integrated context",
            _window_text(integration_memo, max_chars=5000),
            "",
            "## PDF evidence summary",
            _window_text(pdf_summary, max_chars=4500),
            "",
            "## Knowledge graph rows",
            _kg_rows_markdown(kg_records, max_rows=max_kg_rows),
            "",
            "## Analysis note",
            _window_text(analysis_note, max_chars=700),
            "",
            "## Analysis report excerpt",
            "```json",
            excerpt,
            "```",
            "",
            "## Output requirements",
            "- Write **only** the answer text (no JSON).",
            "- **First sentence** must address the user question directly.",
            "- At most two short paragraphs.",
            "- No follow-up questions or suggested tasks.",
            "- If evidence is insufficient, say so in one sentence, then give the best partial answer.",
        ]
    )


def render_knowledge_package_markdown(package: KnowledgePackage) -> str:
    """Render a reader-friendly Knowledge Package for the chat UI."""
    lines: List[str] = [
        "# Knowledge Package",
        "",
        "This package summarizes what the system found and what you can use next.",
        "",
    ]

    if package.clarifying_questions:
        lines.append("## Optional clarifying questions")
        lines.append("These are optional prompts if you want to refine a follow-up query.")
        for q in package.clarifying_questions[:4]:
            lines.append(f"- {q}")
        lines.append("")

    lines.extend(
        [
            "## 1. Bottom-line synthesis",
            package.executive_synthesis or "*(insufficient evidence for a confident synthesis)*",
            "",
            f"Evidence confidence: {package.confidence:.0%} (0% = weak / incomplete; 100% = well supported)",
            "",
            "## 2. Key concepts you can work with",
        ]
    )
    if package.key_concepts:
        for c in package.key_concepts[:10]:
            lines.append(f"- {c}")
    else:
        lines.append("- *(none identified yet)*")

    lines.extend(["", "## 3. Conceptual relationships"])
    if package.conceptual_relationships:
        for rel in package.conceptual_relationships[:10]:
            note = f" ({rel.note})" if rel.note else ""
            src = rel.source or "?"
            tgt = rel.target or "?"
            rel_name = rel.relation or "related_to"
            lines.append(f"- {src} → {tgt} [{rel_name}]{note}")
    else:
        lines.append("- *(none identified yet)*")

    lines.extend(["", "## 4. Evidence highlights"])
    if package.supporting_evidence:
        for e in package.supporting_evidence[:8]:
            lines.append(f"- {e}")
    else:
        lines.append("- *(limited evidence in this run)*")

    lines.extend(["", "## 5. Candidate propositions (for your theorizing)"])
    if package.candidate_propositions:
        for p in package.candidate_propositions[:6]:
            lines.append(f"- {p}")
    else:
        lines.append("- *(none proposed yet)*")

    lines.extend(
        [
            "",
            "## 6. Open questions in the evidence",
            "These are gaps in retrieved literature evidence — useful research opportunities, not errors.",
        ]
    )
    if package.research_gaps:
        for g in package.research_gaps[:6]:
            lines.append(f"- {g}")
    else:
        lines.append("- *(no major gaps flagged)*")

    plain_analyses = []
    for item in package.suggested_next_analyses[:6]:
        text = str(item or "").strip()
        if not text:
            continue
        if text.lower().startswith("follow_up:"):
            plain_analyses.append(text.split(":", 1)[-1].strip())
            continue
        # Keep analysis ids readable without requiring orchestrator helpers here.
        plain_analyses.append(text.replace("_", " "))
    lines.extend(
        [
            "",
            "## 7. If you continue later, useful next probes",
            "These are system-side search/analysis options — you do not run them manually.",
        ]
    )
    if plain_analyses:
        for a in plain_analyses:
            lines.append(f"- {a}")
    else:
        lines.append("- *(none suggested)*")

    lines.extend(
        [
            "",
            "## 8. Suggested readings (vicarious learning)",
            "Use these to build intuition/context — not as a full literature review.",
        ]
    )
    if package.recommended_qualitative_readings:
        for i, r in enumerate(package.recommended_qualitative_readings[:8], 1):
            lines.append(f"{i}. {r}")
    else:
        lines.append("- *(no qualitative readings retrieved in this run)*")

    lines.extend(
        [
            "",
            "## 9. Provenance (where claims came from)",
        ]
    )
    if package.provenance:
        for prov in package.provenance[:12]:
            dois = ", ".join(prov.dois[:6]) if prov.dois else "n/a"
            claim = (prov.claim or "").strip()
            if len(claim) > 220:
                claim = claim[:217].rsplit(" ", 1)[0] + "…"
            lines.append(f"- [{prov.source_type}] {claim} (DOIs: {dois})")
    else:
        lines.append("- *(not attached)*")
    lines.append("")
    return "\n".join(lines).strip()


class IntegrationAgent:

    def __init__(self, client: Any, model: str = "gpt-5.2") -> None:
        self._client = client
        self._model = model

    def integrate(
        self,
        *,
        stage: str,
        question: str,
        pdf_summary: str,
        integration_memo: str,
        kg_records: List[Dict[str, Any]],
        analysis_report: Dict[str, Any] | None,
        analysis_note: str,
        max_kg_rows: int = 12,
    ) -> IntegrationOutput:
        system = "You integrate state across a KG+LLM pipeline. Produce merged context that is faithful to evidence and centered on the user question. Do not invent facts. Keep it concise and useful for the next step."
        user = _build_integrate_user_markdown(
            stage=stage,
            question=question,
            pdf_summary=pdf_summary,
            integration_memo=integration_memo,
            kg_records=kg_records,
            analysis_report=analysis_report,
            analysis_note=analysis_note,
            max_kg_rows=max_kg_rows,
        )
        try:
            return complete_json(
                self._client,
                model=self._model,
                system=system,
                user=user,
                schema_model=IntegrationOutput,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug("Integration LLM returned invalid output: %s", exc)
        except Exception as exc:
            logger.warning("Integration LLM failed: %s: %s", type(exc).__name__, exc)
        merged = (integration_memo or "").strip()
        if pdf_summary.strip():
            merged = f"{merged}\n{pdf_summary.strip()}".strip() if merged else pdf_summary.strip()
        return IntegrationOutput(
            stage=stage,
            merged_context=f'**Question:** {(question or "").strip() or "(not provided)"}\n\n**Best-effort context:**\n{merged}',
            key_points=["integration_fallback"],
            next_focus="",
        )

    def integrate_knowledge(
        self,
        *,
        stage: str,
        question: str,
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        pdf_summary: str,
        prior_state: Optional[Dict[str, Any] | IntegrationState],
        kg_records: List[Dict[str, Any]],
        analysis_report: Dict[str, Any] | None,
        analysis_note: str,
        max_kg_rows: int = 12,
        strict: bool = False,
        cycle_package: Optional[Dict[str, Any] | KnowledgePackage] = None,
    ) -> IntegrationState:
        if isinstance(prior_state, IntegrationState):
            prior = prior_state.model_dump()
        elif isinstance(prior_state, dict):
            prior = prior_state
        else:
            prior = {}
        if isinstance(research_intent, ResearchIntent):
            intent = research_intent.model_dump()
        elif isinstance(research_intent, dict):
            intent = research_intent
        else:
            intent = {}

        # Compact intent to reduce prompt/output size and avoid JSON truncation.
        compact_intent = {
            "objective": (intent.get("objective") or "")[:300],
            "target_concepts": list(intent.get("target_concepts") or [])[:6],
            "discovery_type": intent.get("discovery_type") or "",
            "refined_question": (intent.get("refined_question") or "")[:400],
        }
        compact_prior = {
            "merged_context": (prior.get("merged_context") or "")[:800],
            "emerging_concepts": list(prior.get("emerging_concepts") or [])[:6],
            "propositions": list(prior.get("propositions") or [])[:4],
            "theoretical_gaps": list(prior.get("theoretical_gaps") or [])[:4],
            "confidence": prior.get("confidence", 0),
        }
        if isinstance(cycle_package, KnowledgePackage):
            cycle_pkg = cycle_package.model_dump()
        elif isinstance(cycle_package, dict):
            cycle_pkg = cycle_package
        else:
            cycle_pkg = {}
        compact_cycle = {
            "stage": cycle_pkg.get("stage") or "",
            "executive_synthesis": (cycle_pkg.get("executive_synthesis") or "")[:600],
            "key_concepts": list(cycle_pkg.get("key_concepts") or [])[:6],
            "supporting_evidence": list(cycle_pkg.get("supporting_evidence") or [])[:4],
            "candidate_propositions": list(cycle_pkg.get("candidate_propositions") or [])[:3],
            "research_gaps": list(cycle_pkg.get("research_gaps") or [])[:3],
        }

        system = (
            "You are the Integration Agent in a Knowledge Discovery Ecosystem. "
            "Synthesize the intermediate cycle Knowledge Package plus retrieved KG rows, "
            "Analysaurus analysis, and chunk evidence into an IntegrationState: emerging concepts, "
            "conceptual relationships, theoretical gaps, possible propositions, candidate conceptual "
            "models, conflicting evidence, and confidence. "
            "Be faithful to evidence; do not invent citations. Return JSON only. "
            "Keep output compact: merged_context <= 700 chars; each list <= 5 short items; "
            "key_points <= 5; next_focus <= 200 chars. "
            "conceptual_relationships MUST be objects with keys source, relation, target, note "
            "(not plain strings). Example: "
            "{\"source\":\"perceived security\",\"relation\":\"antecedent_of\",\"target\":\"trust\",\"note\":\"e-commerce\"}."
        )
        user = json.dumps(
            {
                "stage": stage,
                "user_question": question,
                "research_intent": compact_intent,
                "prior_integration_state": compact_prior,
                "cycle_knowledge_package": compact_cycle,
                "pdf_chunk_summary": _window_text(pdf_summary, max_chars=1800),
                "kg_rows": kg_records[: max(1, min(max_kg_rows, 6))],
                "analysis_note": (analysis_note or "")[:500],
                "analysis_report_excerpt": _safe_json(analysis_report, max_chars=2500),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        try:
            out = complete_json(
                self._client,
                model=self._model,
                system=system,
                user=user,
                schema_model=IntegrationState,
            )
            return out.model_copy(update={"stage": out.stage or stage})
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug("IntegrationState LLM returned invalid output: %s", exc)
            if strict:
                raise RuntimeError(f"IntegrationState primary LLM path failed (strict): {exc}") from exc
        except Exception as exc:
            logger.warning("IntegrationState LLM failed: %s: %s", type(exc).__name__, exc)
            if strict:
                raise RuntimeError(f"IntegrationState primary LLM path failed (strict): {exc}") from exc

        # Fallback: lift from legacy integrate + prior fields
        legacy = self.integrate(
            stage=stage,
            question=question,
            pdf_summary=pdf_summary,
            integration_memo=str(prior.get("merged_context") or ""),
            kg_records=kg_records,
            analysis_report=analysis_report,
            analysis_note=analysis_note,
            max_kg_rows=max_kg_rows,
        )
        return IntegrationState(
            stage=stage,
            merged_context=legacy.merged_context,
            key_points=list(legacy.key_points or []) + ["integration_fallback"],
            next_focus=legacy.next_focus or "",
            emerging_concepts=list(prior.get("emerging_concepts") or intent.get("target_concepts") or []),
            conceptual_relationships=[
                ConceptualRelationship.model_validate(r)
                for r in (prior.get("conceptual_relationships") or [])
                if isinstance(r, (dict, ConceptualRelationship))
            ],
            theoretical_gaps=list(prior.get("theoretical_gaps") or []),
            propositions=list(prior.get("propositions") or []),
            candidate_models=list(prior.get("candidate_models") or []),
            conflicting_evidence=list(prior.get("conflicting_evidence") or []),
            confidence=float(prior.get("confidence") or 0.3),
        )

    def build_knowledge_package(
        self,
        *,
        question: str,
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        integration_state: Optional[Dict[str, Any] | IntegrationState],
        evidence_summary: str,
        cited_dois: List[str],
        vicarious: Optional[Dict[str, Any] | VicariousLearningOutput],
        reflection: Optional[Dict[str, Any]],
        last_cypher: Optional[str],
        analysis_note: str = "",
        stage: str = "final",
    ) -> KnowledgePackage:
        if isinstance(research_intent, ResearchIntent):
            intent = research_intent.model_dump()
        elif isinstance(research_intent, dict):
            intent = research_intent
        else:
            intent = {}
        if isinstance(integration_state, IntegrationState):
            integ = integration_state
        elif isinstance(integration_state, dict):
            try:
                integ = IntegrationState.model_validate(integration_state)
            except ValidationError:
                integ = IntegrationState(merged_context=str(integration_state.get("merged_context") or ""))
        else:
            integ = IntegrationState()
        if isinstance(vicarious, VicariousLearningOutput):
            vic = vicarious
        elif isinstance(vicarious, dict):
            try:
                vic = VicariousLearningOutput.model_validate(vicarious)
            except ValidationError:
                vic = VicariousLearningOutput()
        else:
            vic = VicariousLearningOutput()

        readings: List[str] = []
        for seq in vic.reading_sequence:
            if seq:
                readings.append(f"Start with: {seq}")
        for label, bucket in (
            ("Illustrative study", vic.illustrative_studies),
            ("Case study", vic.case_studies),
            ("Narrative / contextual account", vic.narratives),
            ("Practical example", vic.practical_examples),
            ("Contradictory evidence", vic.contradictory_evidence),
        ):
            for item in bucket:
                source = item.doi or item.title_or_label
                if not source:
                    continue
                why = f" — why useful: {item.why_useful}" if item.why_useful else ""
                excerpt = f" — excerpt: {item.excerpt[:160]}" if item.excerpt else ""
                entry = f"{label}: {source}{why}{excerpt}"
                if entry not in readings:
                    readings.append(entry)

        suggested = []
        if isinstance(reflection, dict):
            suggested = list(reflection.get("recommended_analyses") or [])
            if reflection.get("follow_up_question"):
                suggested.append(f"follow_up: {reflection['follow_up_question']}")

        provenance: List[ClaimProvenance] = []
        if integ.key_points:
            provenance.append(
                ClaimProvenance(
                    claim="; ".join(integ.key_points[:5]),
                    source_type="analysis",
                    dois=list(cited_dois)[:12],
                    cypher_ref=(last_cypher or "")[:400],
                )
            )
        if evidence_summary.strip():
            provenance.append(
                ClaimProvenance(
                    claim=evidence_summary.strip()[:500],
                    source_type="chunk",
                    dois=list(cited_dois)[:12],
                )
            )
        if analysis_note.strip():
            provenance.append(
                ClaimProvenance(
                    claim=analysis_note.strip()[:400],
                    source_type="kg",
                    dois=list(cited_dois)[:8],
                    cypher_ref=(last_cypher or "")[:400],
                )
            )

        synthesis = integ.merged_context.strip() or evidence_summary.strip()
        if not synthesis:
            synthesis = f"Limited evidence was available for: {question}"

        return KnowledgePackage(
            stage=stage or "final",
            executive_synthesis=synthesis[:4000],
            key_concepts=list(integ.emerging_concepts or intent.get("target_concepts") or []),
            conceptual_relationships=list(integ.conceptual_relationships or []),
            supporting_evidence=list(integ.key_points or [])
            or ([evidence_summary.strip()[:800]] if evidence_summary.strip() else []),
            candidate_propositions=list(integ.propositions or []),
            research_gaps=list(integ.theoretical_gaps or []),
            suggested_next_analyses=suggested,
            recommended_qualitative_readings=readings[:20],
            provenance=provenance,
            clarifying_questions=list(intent.get("clarifying_questions") or []),
            confidence=float(integ.confidence or 0.0),
        )

    def build_cycle_package_from_extraction(
        self,
        *,
        question: str,
        research_intent: Optional[Dict[str, Any] | ResearchIntent],
        kg_records: List[Dict[str, Any]],
        analysis_report: Dict[str, Any] | None,
        analysis_note: str,
        chunk_summary: str,
        cycle_index: int,
        last_cypher: Optional[str] = None,
    ) -> KnowledgePackage:
        """Intermediate Knowledge Package for one extraction cycle (pre-integration)."""
        if isinstance(research_intent, ResearchIntent):
            intent = research_intent.model_dump()
        elif isinstance(research_intent, dict):
            intent = research_intent
        else:
            intent = {}
        concepts = list(intent.get("target_concepts") or [])[:8]
        central = self._central_constructs_from_analysis(analysis_report, max_items=5)
        for name in central:
            if name not in concepts:
                concepts.append(name)
        evidence: List[str] = []
        if analysis_note.strip():
            evidence.append(analysis_note.strip()[:500])
        if chunk_summary.strip():
            evidence.append(chunk_summary.strip()[:800])
        for row in kg_records[:4]:
            try:
                evidence.append(json.dumps(row, ensure_ascii=False)[:240])
            except (TypeError, ValueError):
                evidence.append(str(row)[:240])
        synthesis_parts = [
            f"Cycle {cycle_index} extraction package for: {question}",
            analysis_note.strip()[:400] if analysis_note.strip() else "",
            chunk_summary.strip()[:600] if chunk_summary.strip() else "",
        ]
        synthesis = "\n".join(p for p in synthesis_parts if p).strip() or f"Cycle {cycle_index} evidence for {question}"
        provenance = []
        if last_cypher or evidence:
            provenance.append(
                ClaimProvenance(
                    claim=synthesis[:400],
                    source_type="kg" if last_cypher else "chunk",
                    dois=[],
                    cypher_ref=(last_cypher or "")[:400],
                )
            )
        return KnowledgePackage(
            stage=f"cycle_{cycle_index}",
            executive_synthesis=synthesis[:2000],
            key_concepts=concepts[:12],
            conceptual_relationships=[],
            supporting_evidence=evidence[:8],
            candidate_propositions=[],
            research_gaps=[],
            suggested_next_analyses=[],
            recommended_qualitative_readings=[],
            provenance=provenance,
            clarifying_questions=list(intent.get("clarifying_questions") or [])[:4],
            confidence=0.25 if evidence else 0.1,
        )

    @staticmethod
    def _central_constructs_from_analysis(analysis_report: Dict[str, Any] | None, *, max_items: int = 5) -> List[str]:
        if not isinstance(analysis_report, dict):
            return []
        l2 = analysis_report.get("level_2_relationship_mining")
        if not isinstance(l2, dict):
            return []
        names: List[str] = []
        for key, _metric in (
            ("indegreecentrality", "Indegree"),
            ("outdegreecentrality", "Outdegree"),
            ("betweennesscentrality", "Betweenness"),
        ):
            node = l2.get(key)
            if not isinstance(node, dict):
                continue
            ranking = node.get("ranking")
            if not isinstance(ranking, list):
                continue
            for row in ranking[:max_items]:
                if not isinstance(row, dict):
                    continue
                concept = row.get("Concept")
                if isinstance(concept, str) and concept.strip():
                    names.append(concept.strip())
        deduped = list(dict.fromkeys(names))
        return deduped[:max_items]

    def compose_final_answer(
        self,
        *,
        question: str,
        pdf_summary: str,
        integration_memo: str,
        kg_records: List[Dict[str, Any]],
        analysis_report: Dict[str, Any] | None,
        analysis_note: str,
        max_kg_rows: int = 10,
    ) -> str:
        system = "You answer exactly the user's question using the evidence sections below. Do not write a generic summary of sources — synthesize into a direct answer. Stay on-topic. Do not ask follow-up questions. If evidence is weak, say so briefly."
        prompt = _build_compose_final_user_markdown(
            question=question,
            integration_memo=integration_memo,
            pdf_summary=pdf_summary,
            kg_records=kg_records,
            analysis_note=analysis_note,
            analysis_report=analysis_report,
            max_kg_rows=max_kg_rows,
        )
        try:
            response = self._client.generate(model=self._model, system=system, prompt=prompt, stream=False)
            text = (response.get("response", "") if isinstance(response, dict) else "").strip()
            if text:
                return text
        except Exception as exc:
            q = (question or "").strip()
            central = self._central_constructs_from_analysis(analysis_report, max_items=5)
            if central:
                return f"**Answer:** The constructs that appear most central in the graph (by centrality rankings) include: {', '.join(central)}.\n\n*(Final-answer LLM call failed: {type(exc).__name__}: {exc})*"
            if pdf_summary.strip():
                return f"**Answer (from PDF evidence; model call failed):** {pdf_summary.strip()[:2000]}\n\n*(Error: {type(exc).__name__}: {exc})*"
            return f"**Question:** {q or '(not provided)'}\n\n**Answer:** Insufficient evidence after an API error ({type(exc).__name__}: {exc})."
        central = self._central_constructs_from_analysis(analysis_report, max_items=5)
        if central:
            return f"Most central constructs in the trust models are: {', '.join(central)}. This is based on graph centrality signals (indegree/outdegree/betweenness) from the KG."
        if pdf_summary.strip():
            return pdf_summary.strip()
        return "Insufficient evidence to answer confidently."
