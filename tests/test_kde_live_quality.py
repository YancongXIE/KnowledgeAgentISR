"""
Live quality tests for KDE agents (requires Azure OpenAI).

Neo4j is optional: Vicarious uses fixture chunks when Neo4j is down.

Run:
    cd "code folder"
    python3 -m pytest tests/test_kde_live_quality.py -v -s
    python3 tests/run_kde_quality_suite.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from kg_agents.integration_agent import render_knowledge_package_markdown
from tests.kde_harness import (
    FIXTURE_CHUNKS,
    FIXTURE_KG_ROWS,
    AgentHarness,
    azure_configured,
    build_harness,
)
from tests.kde_quality import (
    AgentQualityResult,
    evaluate_elicitation,
    evaluate_integration,
    evaluate_knowledge_package,
    evaluate_reflection,
    evaluate_vicarious,
    render_markdown_report,
)

# Collected for report generation
LIVE_RESULTS: List[AgentQualityResult] = []

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "TC01_vague",
        "kind": "vague",
        "question": "trust",
        "expect_vague": True,
    },
    {
        "id": "TC02_definition",
        "kind": "definition",
        "question": "What are the definitions of trust in information systems research?",
        "expect_vague": False,
    },
    {
        "id": "TC03_relationship",
        "kind": "relationship",
        "question": "What antecedents and consequents of trust appear in the MIS Quarterly literature?",
        "expect_vague": False,
    },
    {
        "id": "TC04_research_gap",
        "kind": "research_gap",
        "question": "What theoretical gaps remain around trust in virtual teams?",
        "expect_vague": False,
    },
]


pytestmark = pytest.mark.skipif(not azure_configured(), reason="Azure OpenAI credentials not configured")


@pytest.fixture(scope="module")
def harness() -> AgentHarness:
    h = build_harness(use_neo4j=True)
    yield h
    h.close()


def _record(result: AgentQualityResult) -> AgentQualityResult:
    LIVE_RESULTS.append(result)
    return result


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
class TestElicitationLiveQuality:
    def test_elicitation(self, harness: AgentHarness, case: Dict[str, Any]) -> None:
        intent = harness.elicitation.elicit(case["question"])
        result = _record(
            evaluate_elicitation(
                intent,
                test_case_id=case["id"],
                question=case["question"],
                expect_vague=case["expect_vague"],
            )
        )
        print(f"\n[{result.agent}] {case['id']} score={result.score_pct}% grade={result.grade}")
        assert result.score_pct >= 45, f"Elicitation quality too low: {result.findings}"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
class TestIntegrationLiveQuality:
    def test_integration(self, harness: AgentHarness, case: Dict[str, Any]) -> None:
        intent = harness.elicitation.elicit(case["question"])
        pdf_summary = "\n".join(c["text"][:400] for c in FIXTURE_CHUNKS)
        state = harness.integration.integrate_knowledge(
            stage=f"live_test_{case['id']}",
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            pdf_summary=pdf_summary,
            prior_state=None,
            kg_records=FIXTURE_KG_ROWS,
            analysis_report={
                "level_1_concept_extraction": {
                    "definitions_found": 2,
                    "n_related_publications": 3,
                }
            },
            analysis_note="live quality test with fixture KG rows",
        )
        result = _record(
            evaluate_integration(state, test_case_id=case["id"], question=case["question"])
        )
        print(f"\n[{result.agent}] {case['id']} score={result.score_pct}% grade={result.grade}")
        assert result.score_pct >= 45, f"Integration quality too low: {result.findings}"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
class TestReflectionLiveQuality:
    def test_reflection(self, harness: AgentHarness, case: Dict[str, Any]) -> None:
        intent = harness.elicitation.elicit(case["question"])
        integ = harness.integration.integrate_knowledge(
            stage=f"reflect_{case['id']}",
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            pdf_summary=FIXTURE_CHUNKS[0]["text"],
            prior_state=None,
            kg_records=FIXTURE_KG_ROWS,
            analysis_report=None,
            analysis_note="reflection live test",
        )
        allowed = sorted(harness.orchestrator._allowed_analysis_function_names())
        decision = harness.reflection.reflect(
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            integration_state=integ.model_dump(),
            analysis_brief="L1 defs=2 pubs=3",
            already_run_analyses=["definitions"],
            allowed_analyses=allowed,
            cycle_index=0,
            max_cycles=4,
            kg_excerpt=json.dumps(FIXTURE_KG_ROWS[0])[:300],
            chunk_excerpt=FIXTURE_CHUNKS[0]["text"][:300],
        )
        result = _record(
            evaluate_reflection(
                decision,
                test_case_id=case["id"],
                question=case["question"],
                allowed_analyses=allowed,
                integration_confidence=float(integ.confidence),
            )
        )
        print(f"\n[{result.agent}] {case['id']} score={result.score_pct}% grade={result.grade}")
        assert result.score_pct >= 45, f"Reflection quality too low: {result.findings}"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
class TestVicariousLiveQuality:
    def test_vicarious(self, harness: AgentHarness, case: Dict[str, Any]) -> None:
        intent = harness.elicitation.elicit(case["question"])
        integ = harness.integration.integrate_knowledge(
            stage=f"vicarious_{case['id']}",
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            pdf_summary=FIXTURE_CHUNKS[0]["text"],
            prior_state=None,
            kg_records=FIXTURE_KG_ROWS,
            analysis_report=None,
            analysis_note="vicarious live test",
        )

        if harness.neo4j_ok:
            emb = harness.embed_fn([case["question"]])[0]
            out = harness.vicarious.learn(
                harness.driver,
                question=intent.refined_question or case["question"],
                question_embedding=emb,
                research_intent=intent.model_dump(),
                integration_state=integ.model_dump(),
                fallback_records=FIXTURE_CHUNKS,
            )
            mode = "neo4j_live"
        else:
            with patch.object(
                harness.vicarious._extraction,
                "extract_from_instruction",
                return_value=(FIXTURE_CHUNKS, "fixture chunks (neo4j unavailable)"),
            ):
                out = harness.vicarious.learn(
                    harness.driver,
                    question=intent.refined_question or case["question"],
                    question_embedding=harness.embed_fn([case["question"]])[0],
                    research_intent=intent.model_dump(),
                    integration_state=integ.model_dump(),
                    fallback_records=FIXTURE_CHUNKS,
                )
            mode = "fixture_chunks"

        result = _record(
            evaluate_vicarious(
                out,
                test_case_id=case["id"],
                question=case["question"],
                target_concepts=intent.target_concepts or ["trust"],
            )
        )
        result.warnings.append(f"retrieval_mode={mode}; neo4j_ok={harness.neo4j_ok}")
        print(f"\n[{result.agent}] {case['id']} score={result.score_pct}% grade={result.grade} mode={mode}")
        assert result.score_pct >= 40, f"Vicarious quality too low: {result.findings}"


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["id"] for c in TEST_CASES])
class TestKnowledgePackageLiveQuality:
    def test_knowledge_package(self, harness: AgentHarness, case: Dict[str, Any]) -> None:
        intent = harness.elicitation.elicit(case["question"])
        integ = harness.integration.integrate_knowledge(
            stage=f"package_{case['id']}",
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            pdf_summary="\n".join(c["text"][:300] for c in FIXTURE_CHUNKS),
            prior_state=None,
            kg_records=FIXTURE_KG_ROWS,
            analysis_report=None,
            analysis_note="package live test",
        )
        allowed = sorted(harness.orchestrator._allowed_analysis_function_names())
        reflection = harness.reflection.reflect(
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            integration_state=integ.model_dump(),
            analysis_brief="",
            already_run_analyses=["definitions"],
            allowed_analyses=allowed,
            cycle_index=1,
            max_cycles=4,
        )

        if harness.neo4j_ok:
            vic = harness.vicarious.learn(
                harness.driver,
                question=intent.refined_question or case["question"],
                question_embedding=harness.embed_fn([case["question"]])[0],
                research_intent=intent.model_dump(),
                integration_state=integ.model_dump(),
                fallback_records=FIXTURE_CHUNKS,
            )
        else:
            with patch.object(
                harness.vicarious._extraction,
                "extract_from_instruction",
                return_value=(FIXTURE_CHUNKS, "fixture"),
            ):
                vic = harness.vicarious.learn(
                    harness.driver,
                    question=intent.refined_question or case["question"],
                    question_embedding=harness.embed_fn([case["question"]])[0],
                    research_intent=intent.model_dump(),
                    integration_state=integ.model_dump(),
                )

        package = harness.integration.build_knowledge_package(
            question=intent.refined_question or case["question"],
            research_intent=intent.model_dump(),
            integration_state=integ,
            evidence_summary="\n".join(c["text"][:200] for c in FIXTURE_CHUNKS),
            cited_dois=[c["doi"] for c in FIXTURE_CHUNKS],
            vicarious=vic,
            reflection=reflection.model_dump(),
            last_cypher="MATCH (e:Element) RETURN e LIMIT 5",
            analysis_note="package assembly test",
        )
        md = render_knowledge_package_markdown(package)
        result = _record(
            evaluate_knowledge_package(
                package,
                test_case_id=case["id"],
                question=case["question"],
                expect_clarifying=case["expect_vague"],
            )
        )
        result.warnings.append(f"markdown_length={len(md)}")
        print(f"\n[{result.agent}] {case['id']} score={result.score_pct}% grade={result.grade}")
        assert "# Knowledge Package" in md
        assert result.score_pct >= 45, f"Package quality too low: {result.findings}"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write report after live test session if any results collected."""
    if not LIVE_RESULTS:
        return
    from pathlib import Path

    from tests.kde_harness import azure_configured, probe_neo4j

    neo_ok, neo_detail = probe_neo4j()
    report = render_markdown_report(
        results=LIVE_RESULTS,
        azure_ok=azure_configured(),
        neo4j_ok=neo_ok,
        neo4j_detail=neo_detail,
        test_cases=TEST_CASES,
    )
    out = Path(__file__).resolve().parents[1].parent / "docs" / "KDE_AGENT_TEST_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
