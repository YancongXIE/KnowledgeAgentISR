"""
Quality evaluation helpers for KDE agent live tests.

Scores are heuristic (0–100) for regression tracking, not human ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from kg_agents.models import (
    IntegrationState,
    KnowledgePackage,
    ReflectionDecision,
    ResearchIntent,
    VicariousLearningOutput,
)


@dataclass
class AgentQualityResult:
    agent: str
    test_case_id: str
    question: str
    passed: bool
    score: float
    max_score: float
    grade: str
    findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""
    llm_path: str = "unknown"  # primary | legacy_fallback | fixture_fallback | mock | failed

    @property
    def score_pct(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return round(100.0 * self.score / self.max_score, 1)


def grade_from_pct(pct: float) -> str:
    if pct >= 85:
        return "Good"
    if pct >= 65:
        return "Acceptable"
    if pct >= 45:
        return "Weak"
    return "Poor"


def _keyword_overlap(text: str, keywords: Sequence[str]) -> float:
    if not text or not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / max(len(keywords), 1)


def _add(points: float, max_pts: float, ok: bool, findings: List[str], msg: str) -> float:
    if ok:
        findings.append(f"[+] {msg}")
        return points
    findings.append(f"[-] {msg}")
    return 0.0


def evaluate_elicitation(
    intent: ResearchIntent,
    *,
    test_case_id: str,
    question: str,
    expect_vague: bool = False,
) -> AgentQualityResult:
    findings: List[str] = []
    warnings: List[str] = []
    score = 0.0
    max_score = 100.0

    score += _add(
        20,
        20,
        bool((intent.refined_question or "").strip()),
        findings,
        "refined_question is non-empty",
    )
    score += _add(
        20,
        20,
        bool((intent.objective or "").strip()),
        findings,
        "objective is non-empty",
    )
    score += _add(
        20,
        20,
        bool(intent.target_concepts),
        findings,
        "target_concepts identified",
    )
    score += _add(
        15,
        15,
        bool((intent.discovery_type or "").strip()),
        findings,
        "discovery_type specified",
    )

    q_tokens = [t for t in re.findall(r"[a-zA-Z]{4,}", question.lower())]
    overlap = _keyword_overlap(
        " ".join([intent.refined_question, intent.objective, " ".join(intent.target_concepts)]),
        q_tokens[:5],
    )
    score += _add(15, 15, overlap >= 0.3 or not q_tokens, findings, "output aligned with question keywords")

    if expect_vague:
        ok = bool(intent.clarifying_questions) or intent.is_sufficiently_specified is False
        score += _add(10, 10, ok, findings, "vague input triggers clarifying_questions or underspecified flag")
    else:
        ok = intent.is_sufficiently_specified is not False
        score += _add(10, 10, ok, findings, "specific input marked sufficiently specified")

    pct = 100.0 * score / max_score
    if not intent.theoretical_contribution:
        warnings.append("theoretical_contribution empty (optional but useful for KDE).")
    return AgentQualityResult(
        agent="ElicitationAgent",
        test_case_id=test_case_id,
        question=question,
        passed=pct >= 65,
        score=score,
        max_score=max_score,
        grade=grade_from_pct(pct),
        findings=findings,
        warnings=warnings,
        payload=intent.model_dump(),
        llm_path="primary",
    )


def evaluate_integration(
    state: IntegrationState,
    *,
    test_case_id: str,
    question: str,
) -> AgentQualityResult:
    findings: List[str] = []
    warnings: List[str] = []
    score = 0.0
    max_score = 100.0

    score += _add(25, 25, bool(state.emerging_concepts), findings, "emerging_concepts populated")
    score += _add(
        25,
        25,
        len((state.merged_context or "").strip()) >= 80,
        findings,
        "merged_context substantive (>=80 chars)",
    )
    score += _add(
        15,
        15,
        0.0 <= float(state.confidence) <= 1.0,
        findings,
        "confidence in [0,1]",
    )
    score += _add(
        20,
        20,
        bool(state.propositions or state.theoretical_gaps or state.conceptual_relationships),
        findings,
        "synthesis includes propositions, gaps, or relationships",
    )
    q_tokens = [t for t in re.findall(r"[a-zA-Z]{4,}", question.lower())]
    overlap = _keyword_overlap(
        " ".join(state.emerging_concepts) + " " + (state.merged_context or ""),
        q_tokens[:5],
    )
    score += _add(15, 15, overlap >= 0.2 or not q_tokens, findings, "concepts/context relate to question")

    # Detect legacy fallback signal.
    if "integration_fallback" in (state.key_points or []):
        findings.append("[-] legacy_fallback detected (integration_fallback key_point)")
        score = min(score, 40)
        warnings.append("Result came from legacy IntegrationOutput fallback, not primary IntegrationState LLM.")

    pct = 100.0 * score / max_score
    if state.confidence < 0.35:
        warnings.append("Low confidence — expected when evidence is thin or mocked.")
    return AgentQualityResult(
        agent="IntegrationAgent",
        test_case_id=test_case_id,
        question=question,
        passed=pct >= 65 and "integration_fallback" not in (state.key_points or []),
        score=score,
        max_score=max_score,
        grade=grade_from_pct(pct),
        findings=findings,
        warnings=warnings,
        payload=state.model_dump(),
        llm_path="legacy_fallback" if "integration_fallback" in (state.key_points or []) else "primary",
    )


def evaluate_reflection(
    decision: ReflectionDecision,
    *,
    test_case_id: str,
    question: str,
    allowed_analyses: Sequence[str],
    integration_confidence: float,
) -> AgentQualityResult:
    findings: List[str] = []
    warnings: List[str] = []
    score = 0.0
    max_score = 100.0

    score += _add(25, 25, bool((decision.rationale or "").strip()), findings, "rationale provided")
    score += _add(
        25,
        25,
        all(a in allowed_analyses for a in decision.recommended_analyses) if decision.recommended_analyses else True,
        findings,
        "recommended_analyses within allowed catalog",
    )
    score += _add(
        20,
        20,
        bool(decision.uncertainties) or decision.sufficient,
        findings,
        "uncertainties listed or marked sufficient",
    )

    # Consistency: high confidence integration should not always demand continue
    if integration_confidence >= 0.75 and decision.continue_loop and not decision.recommended_analyses:
        consistent = False
        warnings.append("High integration confidence but continue_loop without next analyses.")
    else:
        consistent = True
    score += _add(15, 15, consistent, findings, "decision consistent with integration confidence")

    score += _add(
        15,
        15,
        bool((decision.follow_up_question or "").strip()) or decision.sufficient,
        findings,
        "follow_up_question or sufficient flag present",
    )

    pct = 100.0 * score / max_score
    return AgentQualityResult(
        agent="ReflectionAgent",
        test_case_id=test_case_id,
        question=question,
        passed=pct >= 65 and "Fallback reflection" not in (decision.rationale or ""),
        score=score,
        max_score=max_score,
        grade=grade_from_pct(pct),
        findings=findings,
        warnings=warnings,
        payload=decision.model_dump(),
        llm_path="heuristic_fallback" if "Fallback reflection" in (decision.rationale or "") else "primary",
    )


def evaluate_vicarious(
    output: VicariousLearningOutput,
    *,
    test_case_id: str,
    question: str,
    target_concepts: Sequence[str],
) -> AgentQualityResult:
    findings: List[str] = []
    warnings: List[str] = []
    score = 0.0
    max_score = 100.0

    buckets = (
        output.illustrative_studies
        + output.case_studies
        + output.narratives
        + output.practical_examples
        + output.contradictory_evidence
    )
    score += _add(25, 25, bool(output.reading_sequence or buckets), findings, "readings or structured items present")
    score += _add(
        25,
        25,
        any((b.excerpt or b.doi or b.title_or_label) for b in buckets) if buckets else False,
        findings,
        "reading items include excerpt, DOI, or label",
    )
    score += _add(
        20,
        20,
        any((b.why_useful or "").strip() for b in buckets) if buckets else bool(output.note),
        findings,
        "why_useful populated or retrieval note present",
    )
    blob = " ".join(
        [b.excerpt + " " + b.why_useful for b in buckets]
        + list(output.reading_sequence)
        + [output.note or ""]
    )
    overlap = _keyword_overlap(blob, list(target_concepts)[:5])
    score += _add(15, 15, overlap >= 0.15 or not target_concepts, findings, "content relates to target concepts")
    score += _add(15, 15, len(output.reading_sequence) >= 1 or len(buckets) >= 1, findings, "suggested reading path exists")

    note = output.note or ""
    primary = note.startswith("llm_primary") or "llm_primary|" in note
    if not primary:
        findings.append("[-] LLM structuring path not confirmed (note missing llm_primary)")
        score = min(score, 40)
        warnings.append("Vicarious output appears to be code packaging fallback, not primary LLM structuring.")

    pct = 100.0 * score / max_score
    if not buckets and output.note:
        warnings.append("Fallback packaging only — check Neo4j retrieval or LLM structuring.")
    return AgentQualityResult(
        agent="VicariousLearningAgent",
        test_case_id=test_case_id,
        question=question,
        passed=pct >= 65 and primary,
        score=score,
        max_score=max_score,
        grade=grade_from_pct(pct),
        findings=findings,
        warnings=warnings,
        payload=output.model_dump(),
        llm_path="primary" if primary else "fixture_fallback",
    )


def evaluate_knowledge_package(
    package: KnowledgePackage,
    *,
    test_case_id: str,
    question: str,
    expect_clarifying: bool = False,
) -> AgentQualityResult:
    findings: List[str] = []
    warnings: List[str] = []
    score = 0.0
    max_score = 100.0

    score += _add(25, 25, len((package.executive_synthesis or "").strip()) >= 40, findings, "executive_synthesis substantive")
    score += _add(15, 15, bool(package.key_concepts), findings, "key_concepts listed")
    score += _add(15, 15, bool(package.provenance), findings, "provenance attached")
    score += _add(
        15,
        15,
        bool(package.research_gaps or package.candidate_propositions),
        findings,
        "gaps or propositions present",
    )
    score += _add(
        15,
        15,
        bool(package.recommended_qualitative_readings) or bool(package.supporting_evidence),
        findings,
        "qualitative readings or supporting evidence present",
    )
    if expect_clarifying:
        score += _add(15, 15, bool(package.clarifying_questions), findings, "clarifying questions in package for vague input")
    else:
        score += _add(15, 15, package.confidence > 0, findings, "confidence recorded")

    pct = 100.0 * score / max_score
    if package.confidence < 0.4:
        warnings.append("Package confidence low — may reflect weak upstream evidence.")
    return AgentQualityResult(
        agent="KnowledgePackage",
        test_case_id=test_case_id,
        question=question,
        passed=pct >= 65,
        score=score,
        max_score=max_score,
        grade=grade_from_pct(pct),
        findings=findings,
        warnings=warnings,
        payload=package.model_dump(),
        llm_path="assembled",
    )


def render_markdown_report(
    *,
    results: List[AgentQualityResult],
    azure_ok: bool,
    neo4j_ok: bool,
    neo4j_detail: str,
    test_cases: List[Dict[str, Any]],
) -> str:
    lines: List[str] = [
        "# KDE Agent Quality Test Report",
        "",
        "Auto-generated report for per-agent evaluation in the Knowledge Discovery Ecosystem.",
        "",
        "## Environment",
        "",
        f"- **Azure OpenAI**: {'available' if azure_ok else 'unavailable (live LLM tests skipped)'}",
        f"- **Neo4j**: {'available' if neo4j_ok else 'unavailable'} — {neo4j_detail}",
        "",
        "## Test cases",
        "",
    ]
    for tc in test_cases:
        lines.append(f"- **{tc['id']}** ({tc['kind']}): `{tc['question']}`")
    lines.extend(["", "## Summary", "", "| Agent | Case | Score | Grade | LLM path | Pass |", "|-------|------|-------|-------|----------|------|"])

    by_agent: Dict[str, List[AgentQualityResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)

    for r in results:
        status = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        lines.append(
            f"| {r.agent} | {r.test_case_id} | {r.score_pct}% | {r.grade} | `{r.llm_path}` | {status} |"
        )

    lines.extend(["", "## Per-agent analysis", ""])
    for agent, agent_results in by_agent.items():
        scores = [x.score_pct for x in agent_results if not x.skipped]
        avg = round(sum(scores) / len(scores), 1) if scores else 0.0
        lines.extend([f"### {agent}", "", f"**Average score (non-skipped):** {avg}%", ""])
        for r in agent_results:
            lines.append(f"#### {r.test_case_id} — {r.grade} ({r.score_pct}%)")
            if r.skipped:
                lines.append(f"- *Skipped:* {r.skip_reason}")
                lines.append("")
                continue
            lines.append(f"- **Question:** {r.question}")
            lines.append(f"- **LLM path:** `{r.llm_path}`")
            for f in r.findings:
                lines.append(f"- {f}")
            for w in r.warnings:
                lines.append(f"- ⚠ {w}")
            lines.append("")
            lines.append("<details><summary>Payload snapshot</summary>")
            lines.append("")
            lines.append("```json")
            import json

            lines.append(json.dumps(r.payload, ensure_ascii=False, indent=2)[:4000])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    lines.extend(
        [
            "## Interpretation guide",
            "",
            "| Grade | Meaning |",
            "|-------|---------|",
            "| Good (≥85%) | Structurally complete and aligned with question intent |",
            "| Acceptable (65–84%) | Usable for KDE demo; minor gaps |",
            "| Weak (45–64%) | Partial output; review prompts or evidence |",
            "| Poor (<45%) | Agent likely failed or returned empty synthesis |",
            "",
            "## How to re-run",
            "",
            "```bash",
            'cd "code folder"',
            "python3 -m pytest tests/test_kde_live_quality.py -v",
            "python3 tests/run_kde_quality_suite.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
