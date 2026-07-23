#!/usr/bin/env python3
"""
Run KDE agent quality suite and write docs/KDE_AGENT_TEST_REPORT.md.

Usage (from code folder/):
    python3 tests/run_kde_quality_suite.py
    python3 tests/run_kde_quality_suite.py --no-llm   # structure-only with mocks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.kde_harness import (  # noqa: E402
    FIXTURE_CHUNKS,
    FIXTURE_KG_ROWS,
    AgentHarness,
    azure_configured,
    build_harness,
    probe_neo4j,
)
from tests.kde_quality import (  # noqa: E402
    AgentQualityResult,
    evaluate_elicitation,
    evaluate_integration,
    evaluate_knowledge_package,
    evaluate_reflection,
    evaluate_vicarious,
    render_markdown_report,
)
from unittest.mock import patch

from tests.conftest import MockLLMClient, mock_payload_for  # noqa: E402
from kg_agents.models import (  # noqa: E402
    IntegrationState,
    ReflectionDecision,
    ResearchIntent,
    VicariousLearningOutput,
)
from kg_agents.elicitation_agent import ElicitationAgent  # noqa: E402
from kg_agents.integration_agent import IntegrationAgent  # noqa: E402
from kg_agents.reflection_agent import ReflectionAgent  # noqa: E402
from kg_agents.vicarious_learning_agent import VicariousLearningAgent  # noqa: E402
from kg_agents.extraction_agent import ExtractionAgent  # noqa: E402


# Re-use test cases from live module
from tests import test_kde_live_quality as live_mod  # noqa: E402

TEST_CASES = live_mod.TEST_CASES


def run_mock_suite() -> list[AgentQualityResult]:
    results: list[AgentQualityResult] = []

    def client_for(case: Dict[str, Any]) -> MockLLMClient:
        q = case["question"]
        concepts = ["trust"] if "trust" in q.lower() else [t for t in q.split() if len(t) > 4][:3]
        return MockLLMClient(
            {
                "ResearchIntent": mock_payload_for(
                    ResearchIntent,
                    objective=f"Investigate: {q[:80]}",
                    target_concepts=concepts or ["trust"],
                    discovery_type={
                        "vague": "relationship",
                        "definition": "definition",
                        "relationship": "relationship",
                        "research_gap": "research_gap",
                    }.get(case["kind"], "relationship"),
                    refined_question=q if len(q) > 12 else f"What does {q} mean in IS research?",
                    is_sufficiently_specified=not case["expect_vague"],
                    clarifying_questions=(
                        ["Which domain of trust do you mean?", "Conceptual or empirical focus?"]
                        if case["expect_vague"]
                        else []
                    ),
                ),
                "IntegrationState": mock_payload_for(
                    IntegrationState,
                    merged_context=(
                        f"Synthesis for {q}: Trust involves vulnerability, risk, and positive expectations "
                        f"grounded in fixture publications."
                    ),
                    emerging_concepts=concepts or ["trust"],
                    propositions=["Trust reduces perceived risk in online settings."],
                    theoretical_gaps=["Limited qualitative evidence on virtual teams."],
                    confidence=0.72,
                ),
                "ReflectionDecision": mock_payload_for(
                    ReflectionDecision,
                    sufficient=case["kind"] == "definition",
                    continue_loop=case["kind"] != "definition",
                    recommended_analyses=["antecedents_consequents", "definitions"],
                    rationale=f"Evidence review for {case['kind']} question.",
                    uncertainties=["Coverage of peripheral constructs may be incomplete."],
                    follow_up_question="Which consequents matter most for the focal concept?",
                ),
                "VicariousLearningOutput": mock_payload_for(
                    VicariousLearningOutput,
                    illustrative_studies=[
                        {
                            "title_or_label": "Qualitative trust study",
                            "doi": "10.2307/qualitative-trust-001",
                            "why_useful": "Provides contextual immersion into trust formation.",
                            "excerpt": FIXTURE_CHUNKS[0]["text"][:200],
                        }
                    ],
                    reading_sequence=["10.2307/qualitative-trust-001"],
                    note="mock vicarious structuring",
                ),
            }
        )

    for case in TEST_CASES:
        q = case["question"]
        client = client_for(case)
        elicitation = ElicitationAgent(client)
        integration = IntegrationAgent(client)
        reflection = ReflectionAgent(client)
        extraction = ExtractionAgent(lambda t: [[0.1] * 8 for _ in t])
        vicarious = VicariousLearningAgent(client, extraction)
        intent = elicitation.elicit(q)
        results.append(
            evaluate_elicitation(intent, test_case_id=case["id"], question=q, expect_vague=case["expect_vague"])
        )
        integ = integration.integrate_knowledge(
            stage="mock",
            question=q,
            research_intent=intent.model_dump(),
            pdf_summary=FIXTURE_CHUNKS[0]["text"],
            prior_state=None,
            kg_records=FIXTURE_KG_ROWS,
            analysis_report=None,
            analysis_note="mock",
        )
        results.append(evaluate_integration(integ, test_case_id=case["id"], question=q))
        decision = reflection.reflect(
            question=q,
            research_intent=intent.model_dump(),
            integration_state=integ.model_dump(),
            analysis_brief="",
            already_run_analyses=[],
            allowed_analyses=["definitions", "antecedents_consequents"],
            cycle_index=0,
            max_cycles=4,
        )
        results.append(
            evaluate_reflection(
                decision,
                test_case_id=case["id"],
                question=q,
                allowed_analyses=["definitions", "antecedents_consequents"],
                integration_confidence=integ.confidence,
            )
        )
        with patch.object(extraction, "extract_from_instruction", return_value=(FIXTURE_CHUNKS, "mock")):
            vic = vicarious.learn(
                None,
                question=q,
                question_embedding=[0.1] * 8,
                research_intent=intent.model_dump(),
                integration_state=integ.model_dump(),
            )
        results.append(
            evaluate_vicarious(vic, test_case_id=case["id"], question=q, target_concepts=["trust"])
        )
        pkg = integration.build_knowledge_package(
            question=q,
            research_intent=intent.model_dump(),
            integration_state=integ,
            evidence_summary=FIXTURE_CHUNKS[0]["text"],
            cited_dois=["10.2307/qualitative-trust-001"],
            vicarious=vic,
            reflection=decision.model_dump(),
            last_cypher=None,
        )
        results.append(
            evaluate_knowledge_package(
                pkg, test_case_id=case["id"], question=q, expect_clarifying=case["expect_vague"]
            )
        )
    return results


def run_live_suite(*, strict: bool = True) -> list[AgentQualityResult]:
    """Run live Azure LLM tests. When strict=True, agent fallbacks raise and are recorded as failed."""
    results: list[AgentQualityResult] = []
    harness = build_harness(use_neo4j=True)
    try:
        allowed = sorted(harness.orchestrator._allowed_analysis_function_names())
        for case in TEST_CASES:
            q = case["question"]
            print(f"\n=== {case['id']} (strict={strict}) ===")

            # --- Elicitation ---
            try:
                intent = harness.elicitation.elicit(q, strict=strict)
                r = evaluate_elicitation(
                    intent, test_case_id=case["id"], question=q, expect_vague=case["expect_vague"]
                )
                r.llm_path = "primary"
            except Exception as exc:
                r = AgentQualityResult(
                    agent="ElicitationAgent",
                    test_case_id=case["id"],
                    question=q,
                    passed=False,
                    score=0,
                    max_score=100,
                    grade="Poor",
                    findings=[f"[-] primary LLM failed: {exc}"],
                    llm_path="failed",
                )
                results.append(r)
                print(f"Elicitation FAILED: {exc}")
                continue
            results.append(r)
            print(f"Elicitation {r.score_pct}% path={r.llm_path}")

            # --- Integration ---
            try:
                integ = harness.integration.integrate_knowledge(
                    stage=f"suite_{case['id']}",
                    question=intent.refined_question or q,
                    research_intent=intent.model_dump(),
                    pdf_summary="\n".join(c["text"] for c in FIXTURE_CHUNKS),
                    prior_state=None,
                    kg_records=FIXTURE_KG_ROWS,
                    analysis_report={"level_1_concept_extraction": {"definitions_found": 2}},
                    analysis_note="suite",
                    strict=strict,
                )
                r = evaluate_integration(integ, test_case_id=case["id"], question=q)
            except Exception as exc:
                r = AgentQualityResult(
                    agent="IntegrationAgent",
                    test_case_id=case["id"],
                    question=q,
                    passed=False,
                    score=0,
                    max_score=100,
                    grade="Poor",
                    findings=[f"[-] primary LLM failed: {exc}"],
                    llm_path="failed",
                )
                results.append(r)
                print(f"Integration FAILED: {exc}")
                continue
            results.append(r)
            print(f"Integration {r.score_pct}% path={r.llm_path}")

            # --- Reflection ---
            try:
                decision = harness.reflection.reflect(
                    question=intent.refined_question or q,
                    research_intent=intent.model_dump(),
                    integration_state=integ.model_dump(),
                    analysis_brief="L1 defs=2",
                    already_run_analyses=["definitions"],
                    allowed_analyses=allowed,
                    cycle_index=0,
                    max_cycles=4,
                    strict=strict,
                )
                r = evaluate_reflection(
                    decision,
                    test_case_id=case["id"],
                    question=q,
                    allowed_analyses=allowed,
                    integration_confidence=integ.confidence,
                )
            except Exception as exc:
                r = AgentQualityResult(
                    agent="ReflectionAgent",
                    test_case_id=case["id"],
                    question=q,
                    passed=False,
                    score=0,
                    max_score=100,
                    grade="Poor",
                    findings=[f"[-] primary LLM failed: {exc}"],
                    llm_path="failed",
                )
                results.append(r)
                print(f"Reflection FAILED: {exc}")
                continue
            results.append(r)
            print(f"Reflection {r.score_pct}% path={r.llm_path}")

            # --- Vicarious (fixture passages as INPUT only; LLM must structure) ---
            try:
                if harness.neo4j_ok:
                    vic = harness.vicarious.learn(
                        harness.driver,
                        question=intent.refined_question or q,
                        question_embedding=harness.embed_fn([q])[0],
                        research_intent=intent.model_dump(),
                        integration_state=integ.model_dump(),
                        fallback_records=FIXTURE_CHUNKS,
                        strict=strict,
                    )
                    retrieval_mode = "neo4j"
                else:
                    vic = harness.vicarious.learn(
                        harness.driver,
                        question=intent.refined_question or q,
                        question_embedding=harness.embed_fn([q])[0],
                        research_intent=intent.model_dump(),
                        integration_state=integ.model_dump(),
                        passages=FIXTURE_CHUNKS,
                        strict=strict,
                    )
                    retrieval_mode = "fixture_passages_input"
                r = evaluate_vicarious(
                    vic,
                    test_case_id=case["id"],
                    question=q,
                    target_concepts=intent.target_concepts or ["trust"],
                )
                r.warnings.append(f"retrieval_mode={retrieval_mode}; neo4j_ok={harness.neo4j_ok}")
            except Exception as exc:
                r = AgentQualityResult(
                    agent="VicariousLearningAgent",
                    test_case_id=case["id"],
                    question=q,
                    passed=False,
                    score=0,
                    max_score=100,
                    grade="Poor",
                    findings=[f"[-] primary LLM failed: {exc}"],
                    warnings=[f"neo4j_ok={harness.neo4j_ok}"],
                    llm_path="failed",
                )
                results.append(r)
                print(f"Vicarious FAILED: {exc}")
                continue
            results.append(r)
            print(f"Vicarious {r.score_pct}% path={r.llm_path}")

            # --- Knowledge Package (assembly from primary LLM outputs) ---
            pkg = harness.integration.build_knowledge_package(
                question=intent.refined_question or q,
                research_intent=intent.model_dump(),
                integration_state=integ,
                evidence_summary=FIXTURE_CHUNKS[0]["text"],
                cited_dois=[c["doi"] for c in FIXTURE_CHUNKS],
                vicarious=vic,
                reflection=decision.model_dump(),
                last_cypher="MATCH (n) RETURN n LIMIT 1",
            )
            r = evaluate_knowledge_package(
                pkg, test_case_id=case["id"], question=q, expect_clarifying=case["expect_vague"]
            )
            r.llm_path = "assembled_from_primary"
            results.append(r)
            print(f"Package {r.score_pct}% path={r.llm_path}")
    finally:
        harness.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="KDE agent quality suite")
    parser.add_argument("--no-llm", action="store_true", help="Run mock-only structural quality suite")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow agent fallbacks (legacy/heuristic). Default is strict primary-LLM-only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "docs" / "KDE_AGENT_TEST_REPORT.md",
        help="Report output path",
    )
    args = parser.parse_args()

    neo_ok, neo_detail = probe_neo4j()
    if args.no_llm or not azure_configured():
        print("Running MOCK quality suite (no Azure LLM).")
        results = run_mock_suite()
        azure_ok = False
    else:
        strict = not args.allow_fallback
        print(f"Running LIVE quality suite (Azure LLM, strict={strict}).")
        if not neo_ok:
            print(
                f"Neo4j unavailable — Vicarious uses fixture passages as INPUT only; "
                f"LLM structuring still required. ({neo_detail})"
            )
        results = run_live_suite(strict=strict)
        azure_ok = True

    report = render_markdown_report(
        results=results,
        azure_ok=azure_ok,
        neo4j_ok=neo_ok,
        neo4j_detail=neo_detail,
        test_cases=TEST_CASES,
    )
    # Prepend mode banner
    mode_line = (
        "# Mode: STRICT primary-LLM-only (no agent fallbacks)\n\n"
        if (azure_ok and not args.allow_fallback and not args.no_llm)
        else "# Mode: mock or allow-fallback\n\n"
    )
    report = mode_line + report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")

    passed = sum(1 for r in results if r.passed and not r.skipped)
    total = sum(1 for r in results if not r.skipped)
    primary = sum(1 for r in results if r.llm_path in {"primary", "assembled_from_primary"} and not r.skipped)
    failed_path = sum(1 for r in results if r.llm_path == "failed")
    print(f"Passed: {passed}/{total}")
    print(f"Primary LLM path: {primary}/{total}; Failed path: {failed_path}")
    return 0 if passed == total and failed_path == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())