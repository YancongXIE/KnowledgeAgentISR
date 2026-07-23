"""
Smoke tests for KDE agents (Elicitation, Integration, Reflection, Vicarious).

Run from `code folder/`:

    pip install pytest
    python -m pytest tests/test_kde_agents.py -v

No Azure or Neo4j required — LLM and Neo4j calls are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kg_agents.elicitation_agent import ElicitationAgent
from kg_agents.integration_agent import IntegrationAgent, render_knowledge_package_markdown
from kg_agents.models import (
    IntegrationState,
    KnowledgePackage,
    ReflectionDecision,
    ResearchIntent,
    VicariousLearningOutput,
    VicariousReadingItem,
)
from kg_agents.orchestrator import KGMultiAgentOrchestrator
from kg_agents.reflection_agent import ReflectionAgent
from kg_agents.vicarious_learning_agent import VicariousLearningAgent
from tests.conftest import MockLLMClient, mock_payload_for


# ---------------------------------------------------------------------------
# ElicitationAgent
# ---------------------------------------------------------------------------


class TestElicitationAgent:
    def test_empty_question_returns_clarifying_questions(self):
        agent = ElicitationAgent(MockLLMClient())
        intent = agent.elicit("   ")
        assert intent.is_sufficiently_specified is False
        assert intent.clarifying_questions
        assert intent.refined_question == ""

    def test_llm_success_populates_research_intent(self):
        payload = mock_payload_for(
            ResearchIntent,
            objective="Define trust in IS",
            target_concepts=["trust"],
            discovery_type="definition",
            is_sufficiently_specified=True,
            refined_question="What are trust definitions in IS research?",
            clarifying_questions=[],
        )
        client = MockLLMClient({"ResearchIntent": payload})
        agent = ElicitationAgent(client)
        intent = agent.elicit("trust")
        assert intent.objective == "Define trust in IS"
        assert "trust" in intent.target_concepts
        assert intent.refined_question

    def test_llm_failure_uses_fallback(self):
        client = MockLLMClient(fail=True)
        agent = ElicitationAgent(client)
        intent = agent.elicit("What is trust?")
        assert intent.refined_question == "What is trust?"
        assert intent.is_sufficiently_specified is True

    def test_missing_refined_question_filled_from_input(self):
        payload = mock_payload_for(ResearchIntent, refined_question="", objective="x")
        client = MockLLMClient({"ResearchIntent": payload})
        agent = ElicitationAgent(client)
        intent = agent.elicit("Original question")
        assert intent.refined_question == "Original question"


# ---------------------------------------------------------------------------
# IntegrationAgent
# ---------------------------------------------------------------------------


class TestIntegrationAgent:
    def test_integrate_knowledge_returns_integration_state(self):
        payload = mock_payload_for(
            IntegrationState,
            stage="cycle_0",
            merged_context="Trust involves vulnerability.",
            emerging_concepts=["trust", "risk"],
            propositions=["Trust reduces perceived risk."],
            theoretical_gaps=["Limited cross-domain evidence"],
            confidence=0.72,
        )
        client = MockLLMClient({"IntegrationState": payload})
        agent = IntegrationAgent(client)
        out = agent.integrate_knowledge(
            stage="cycle_0",
            question="What is trust?",
            research_intent={"target_concepts": ["trust"]},
            pdf_summary="Trust is willingness to be vulnerable.",
            prior_state=None,
            kg_records=[{"elementName": "trust"}],
            analysis_report=None,
            analysis_note="test",
        )
        assert out.emerging_concepts == ["trust", "risk"]
        assert out.confidence == pytest.approx(0.72)
        assert "vulnerability" in out.merged_context

    def test_build_knowledge_package_assembles_sections(self):
        agent = IntegrationAgent(MockLLMClient())
        pkg = agent.build_knowledge_package(
            question="What is trust?",
            research_intent={"clarifying_questions": ["Which trust domain?"]},
            integration_state=IntegrationState(
                merged_context="Synthesis about trust.",
                emerging_concepts=["trust"],
                propositions=["Trust relates to vulnerability."],
                theoretical_gaps=["Need more qualitative evidence"],
                confidence=0.65,
            ),
            evidence_summary="Several definitions cite vulnerability.",
            cited_dois=["10.2307/1234567"],
            vicarious=VicariousLearningOutput(
                reading_sequence=["10.2307/1234567"],
                illustrative_studies=[
                    VicariousReadingItem(
                        title_or_label="Case study A",
                        doi="10.2307/1234567",
                        why_useful="Qualitative immersion",
                    )
                ],
            ),
            reflection={"recommended_analyses": ["antecedents_consequents"]},
            last_cypher="MATCH (n) RETURN n LIMIT 1",
            analysis_note="graph analysis ok",
        )
        assert pkg.executive_synthesis
        assert "trust" in pkg.key_concepts
        assert pkg.candidate_propositions
        assert pkg.research_gaps
        assert pkg.suggested_next_analyses == ["antecedents_consequents"]
        assert pkg.clarifying_questions == ["Which trust domain?"]
        assert pkg.provenance

    def test_render_knowledge_package_markdown(self):
        pkg = KnowledgePackage(
            executive_synthesis="Trust is central.",
            key_concepts=["trust"],
            confidence=0.8,
        )
        md = render_knowledge_package_markdown(pkg)
        assert "# Knowledge Package" in md
        assert "Bottom-line synthesis" in md
        assert "Trust is central." in md
        assert "trust" in md
        assert "Suggested readings" in md or "Key concepts" in md


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


class TestReflectionAgent:
    def test_reflect_continue_loop_when_not_sufficient(self):
        payload = mock_payload_for(
            ReflectionDecision,
            sufficient=False,
            continue_loop=True,
            recommended_analyses=["antecedents_consequents"],
            rationale="Need relationship evidence.",
            uncertainties=["Definition coverage incomplete"],
        )
        client = MockLLMClient({"ReflectionDecision": payload})
        agent = ReflectionAgent(client)
        decision = agent.reflect(
            question="What is trust?",
            research_intent={"refined_question": "What is trust?"},
            integration_state={"confidence": 0.3, "emerging_concepts": ["trust"]},
            analysis_brief="L1 defs=1",
            already_run_analyses=["definitions"],
            allowed_analyses=["definitions", "antecedents_consequents", "related_publications"],
            cycle_index=0,
            max_cycles=4,
        )
        assert decision.continue_loop is True
        assert "antecedents_consequents" in decision.recommended_analyses

    def test_reflect_stops_when_exhausted_cycles(self):
        payload = mock_payload_for(
            ReflectionDecision,
            sufficient=False,
            continue_loop=True,  # LLM wants more, but cycles exhausted
            recommended_analyses=["betweennesscentrality"],
        )
        client = MockLLMClient({"ReflectionDecision": payload})
        agent = ReflectionAgent(client)
        decision = agent.reflect(
            question="What is trust?",
            research_intent={},
            integration_state={"confidence": 0.2},
            analysis_brief="",
            already_run_analyses=["definitions", "related_publications"],
            allowed_analyses=["definitions", "betweennesscentrality"],
            cycle_index=3,
            max_cycles=4,
        )
        assert decision.continue_loop is False
        assert decision.sufficient is True

    def test_reflect_fallback_on_llm_failure(self):
        client = MockLLMClient(fail=True)
        agent = ReflectionAgent(client)
        decision = agent.reflect(
            question="Q",
            research_intent={},
            integration_state={"confidence": 0.3},
            analysis_brief="",
            already_run_analyses=[],
            allowed_analyses=["definitions"],
            cycle_index=0,
            max_cycles=4,
        )
        assert decision.continue_loop is True
        assert decision.uncertainties


# ---------------------------------------------------------------------------
# VicariousLearningAgent
# ---------------------------------------------------------------------------


class TestVicariousLearningAgent:
    def test_learn_structures_retrieved_passages(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_from_instruction.return_value = (
            [
                {
                    "text": "We conducted an ethnographic study of trust in virtual teams.",
                    "doi": "10.2307/9999999",
                    "score": 0.91,
                }
            ],
            "mock vector retrieval",
        )
        payload = mock_payload_for(
            VicariousLearningOutput,
            illustrative_studies=[
                {
                    "title_or_label": "Ethnography of trust",
                    "doi": "10.2307/9999999",
                    "why_useful": "Qualitative immersion",
                    "excerpt": "ethnographic study",
                }
            ],
            reading_sequence=["10.2307/9999999"],
            note="structured by LLM",
        )
        client = MockLLMClient({"VicariousLearningOutput": payload})
        agent = VicariousLearningAgent(client, mock_extraction)
        driver = MagicMock()
        out = agent.learn(
            driver,
            question="How is trust experienced qualitatively?",
            question_embedding=[0.1] * 8,
            research_intent={"target_concepts": ["trust"]},
            integration_state={"emerging_concepts": ["trust"], "propositions": ["Trust is relational."]},
        )
        assert out.reading_sequence
        assert out.illustrative_studies or out.note
        mock_extraction.extract_from_instruction.assert_called_once()

    def test_learn_fallback_when_llm_fails(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_from_instruction.return_value = (
            [{"text": "Case study excerpt about trust.", "doi": "10.1/abc"}],
            "mock retrieval",
        )
        client = MockLLMClient(fail=True)
        agent = VicariousLearningAgent(client, mock_extraction)
        out = agent.learn(
            MagicMock(),
            question="trust case studies",
            question_embedding=None,
            research_intent=None,
            integration_state=None,
        )
        assert out.reading_sequence or out.illustrative_studies or out.narratives


# ---------------------------------------------------------------------------
# Orchestrator wiring (graph nodes & routing)
# ---------------------------------------------------------------------------


def _make_orchestrator(client: MockLLMClient, *, max_cycles: int = 4) -> KGMultiAgentOrchestrator:
    driver = MagicMock()
    embed = lambda texts: [[0.0] * 8 for _ in texts]
    orch = KGMultiAgentOrchestrator(
        driver,
        client,
        embed,
        model="test-model",
        max_cycles=max_cycles,
        use_kg=True,
        log_progress=False,
    )
    orch.internalization.retrieve = MagicMock(return_value=[])
    orch.internalization.persist = MagicMock(return_value={"ok": True, "uuid": "mem-test"})
    return orch


class TestOrchestratorKDEWiring:
    def test_effective_question_prefers_refined_question(self):
        orch = _make_orchestrator(MockLLMClient())
        state = {
            "question": "trust",
            "research_intent": {"refined_question": "What are trust definitions in IS?"},
        }
        assert orch._effective_question(state) == "What are trust definitions in IS?"

    def test_node_elicit_writes_research_intent(self):
        payload = mock_payload_for(
            ResearchIntent,
            refined_question="Refined trust question",
            target_concepts=["trust"],
            discovery_type="definition",
        )
        orch = _make_orchestrator(MockLLMClient({"ResearchIntent": payload}))
        out = orch._node_elicit({"question": "trust", "scratchpad": [], "iteration": 0})
        assert out["research_intent"]["refined_question"] == "Refined trust question"
        assert any("[elicitation]" in line for line in out["scratchpad"])

    def test_route_after_checkpoint_to_vicarious_when_stopping(self):
        orch = _make_orchestrator(MockLLMClient())
        assert orch._route_after_checkpoint({"stop_gathering": True}) == "vicarious_learning"
        assert orch._route_after_checkpoint({"stop_gathering": False}) == "select_analysis"
        assert (
            orch._route_after_checkpoint({"await_human_gate": True, "pause_status": "awaiting_human_feedback"})
            == "pause_exit"
        )

    def test_checkpoint_reflection_sets_recommended_analyses(self):
        reflect_payload = mock_payload_for(
            ReflectionDecision,
            sufficient=False,
            continue_loop=True,
            recommended_analyses=["antecedents_consequents"],
        )
        integ_payload = mock_payload_for(
            IntegrationState,
            merged_context="memo",
            emerging_concepts=["trust"],
            confidence=0.4,
        )
        client = MockLLMClient(
            {
                "ReflectionDecision": reflect_payload,
                "IntegrationState": integ_payload,
            }
        )
        orch = _make_orchestrator(client, max_cycles=4)
        with patch.object(orch.integration, "integrate_knowledge") as mock_integ:
            mock_integ.return_value = IntegrationState.model_validate(integ_payload)
            state = {
                "question": "What is trust?",
                "research_intent": {"refined_question": "What is trust?"},
                "analysis_step": 0,
                "analysis_queue": ["definitions"],
                "extracted_records": [],
                "last_records": [],
                "integration_trace": [],
                "scratchpad": [],
                "iteration": 0,
            }
            out = orch._node_checkpoint(state)
        assert out["reflection"]["recommended_analyses"] == ["antecedents_consequents"]
        assert out["recommended_analyses"] == ["antecedents_consequents"]
        assert out["stop_gathering"] is False
        assert out["analysis_step"] == 1

    def test_select_analysis_prefers_reflection_recommendations(self):
        orch = _make_orchestrator(MockLLMClient(fail=True))
        state = {
            "question": "trust",
            "research_intent": {},
            "analysis_step": 1,
            "analysis_queue": ["definitions"],
            "recommended_analyses": ["antecedents_consequents"],
            "last_records": [],
            "extracted_records": [],
            "integration_memo": "",
            "analysis_report": None,
        }
        chosen = orch._select_next_analysis(state)
        assert chosen == "antecedents_consequents"

    def test_build_knowledge_package_node_sets_final_answer(self):
        orch = _make_orchestrator(MockLLMClient())
        state = {
            "question": "What is trust?",
            "research_intent": {"clarifying_questions": []},
            "integration_state": IntegrationState(
                merged_context="Trust synthesis.",
                emerging_concepts=["trust"],
                confidence=0.7,
            ).model_dump(),
            "evidence_summary": "Evidence line.",
            "summary_cited_dois": [],
            "scratchpad": [],
            "iteration": 0,
        }
        out = orch._node_build_knowledge_package(state)
        assert out["knowledge_package"]["executive_synthesis"]
        assert out["final_answer"]
        assert "# Knowledge Package" in out["final_answer"]

    def test_graph_has_kde_nodes(self):
        orch = _make_orchestrator(MockLLMClient())
        # Compiled graph should include new KDE nodes (smoke).
        node_names = set(orch._graph.nodes.keys())
        for required in (
            "elicit",
            "checkpoint",
            "vicarious_learning",
            "build_cycle_knowledge_package",
            "build_knowledge_package",
            "pause_exit",
        ):
            assert required in node_names

    def test_route_after_elicit_pauses_when_interactive_underspecified(self):
        orch = _make_orchestrator(MockLLMClient())
        assert orch._route_after_elicit({"pause_status": "needs_clarification"}) == "pause_exit"
        assert orch._route_after_elicit({"pause_status": None}) == "interpret"

    def test_build_cycle_knowledge_package_node(self):
        orch = _make_orchestrator(MockLLMClient())
        state = {
            "question": "What is trust?",
            "research_intent": {"target_concepts": ["trust"], "refined_question": "What is trust?"},
            "analysis_step": 0,
            "last_records": [{"elementName": "trust"}],
            "analysis_report": None,
            "analysis_note": "Found related pubs.",
            "extracted_records": [{"text": "Trust is ...", "doi": "10.1/x"}],
            "cycle_knowledge_packages": [],
            "scratchpad": [],
            "iteration": 0,
        }
        out = orch._node_build_cycle_knowledge_package(state)
        assert out["current_cycle_package"]["stage"] == "cycle_0"
        assert len(out["cycle_knowledge_packages"]) == 1
        assert "trust" in out["current_cycle_package"]["key_concepts"]

    def test_interactive_checkpoint_sets_human_gate(self):
        reflect_payload = mock_payload_for(
            ReflectionDecision,
            sufficient=False,
            continue_loop=True,
            recommended_analyses=["antecedents_consequents"],
            rationale="Need more antecedents.",
            follow_up_question="Which antecedents matter?",
        )
        orch = _make_orchestrator(MockLLMClient({"ReflectionDecision": reflect_payload}), max_cycles=4)
        state = {
            "question": "What is trust?",
            "research_intent": {"refined_question": "What is trust?"},
            "analysis_step": 0,
            "analysis_queue": ["definitions"],
            "extracted_records": [],
            "last_records": [],
            "integration_trace": [],
            "scratchpad": [],
            "iteration": 0,
            "interactive": True,
            "integration_state": {"emerging_concepts": ["trust"], "theoretical_gaps": ["context"]},
        }
        out = orch._node_checkpoint(state)
        assert out["pause_status"] == "awaiting_human_feedback"
        assert out["await_human_gate"] is True
        prompt = out["collaboration_prompt"] or ""
        assert "What I need from you" in prompt
        assert "continue" in prompt.lower()
        assert "stop" in prompt.lower()
        assert "Rationale:" not in prompt
        assert "Recommended next analyses:" not in prompt
        assert orch._route_after_checkpoint(out) == "pause_exit"


class TestSessionStoreAndInternalization:
    def test_session_store_roundtrip(self):
        from kg_agents.session_store import SessionStore

        store = SessionStore(ttl_seconds=60)
        session = store.create(
            phase="elicitation",
            graph_state={"question": "trust", "interactive": True},
            clarifying_questions=["What concept?"],
        )
        got = store.get(session.session_id)
        assert got is not None
        assert got.phase == "elicitation"
        popped = store.pop(session.session_id)
        assert popped is not None
        assert store.get(session.session_id) is None

    def test_internalization_persist_and_retrieve(self):
        from kg_agents.internalization_agent import InternalizationAgent

        stored: dict = {}

        class FakeResult:
            def __init__(self, rows):
                self._rows = rows

            def __iter__(self):
                return iter(self._rows)

            def single(self):
                return self._rows[0] if self._rows else None

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def run(self, cypher, **params):
                if "CREATE" in cypher:
                    stored[params["uuid"]] = params
                    return FakeResult([{"uuid": params["uuid"]}])
                rows = []
                for uid, p in stored.items():
                    rows.append(
                        {
                            "uuid": uid,
                            "text": p["text"],
                            "executive_synthesis": p["synthesis"],
                            "concepts": p["concepts"],
                            "question": p["question"],
                            "created_at": p["created_at"],
                            "embedding": p["embedding"],
                        }
                    )
                return FakeResult(rows)

        driver = MagicMock()
        driver.session.return_value = FakeSession()
        agent = InternalizationAgent(driver)
        pkg = KnowledgePackage(
            stage="final",
            executive_synthesis="Trust is relational.",
            key_concepts=["trust"],
            confidence=0.8,
        )
        result = agent.persist(package=pkg, embedding=[1.0, 0.0, 0.0], question="What is trust?")
        assert result["ok"] is True
        hits = agent.retrieve(question_embedding=[1.0, 0.0, 0.0], top_k=2, query_text="trust")
        assert hits
        assert "Trust" in hits[0]["executive_synthesis"] or "trust" in hits[0]["text"].lower()

    def test_continue_session_from_elicitation(self):
        from kg_agents.session_store import SessionStore

        intent_payload = mock_payload_for(
            ResearchIntent,
            refined_question="Trust antecedents in e-commerce",
            target_concepts=["trust"],
            discovery_type="relationship",
            is_sufficiently_specified=True,
            clarifying_questions=[],
        )
        client = MockLLMClient({"ResearchIntent": intent_payload})
        store = SessionStore(ttl_seconds=60)
        orch = _make_orchestrator(client)
        orch._session_store = store

        # Seed paused elicitation session without running full graph.
        session = store.create(
            phase="elicitation",
            graph_state={
                "question": "trust?",
                "scratchpad": [],
                "iteration": 1,
                "interactive": True,
                "use_kg": True,
                "analysis_step": 0,
                "analysis_queue": [],
                "extracted_records": [],
                "last_records": [],
                "integration_memo": "",
                "integration_trace": [],
                "cycle_knowledge_packages": [],
                "agent_memories": [],
                "research_intent": {
                    "is_sufficiently_specified": False,
                    "clarifying_questions": ["Which domain?"],
                    "refined_question": "trust?",
                },
            },
            clarifying_questions=["Which domain?"],
            interactive=True,
            use_kg=True,
        )

        # Patch graph invoke to avoid full pipeline; just return resumed interpret-ready state.
        def fake_invoke(state):
            state = dict(state)
            state["pause_status"] = None
            state["final_answer"] = "Resumed answer"
            state["knowledge_package"] = {"executive_synthesis": "ok", "stage": "final"}
            state["iteration"] = int(state.get("iteration") or 0) + 1
            return state

        orch._graph.invoke = fake_invoke  # type: ignore[method-assign]
        result = orch.continue_session(
            session_id=session.session_id,
            clarification_answers="Focus on e-commerce trust.",
        )
        assert result.status == "complete"
        assert result.state.research_intent is not None
        assert result.state.research_intent.get("is_sufficiently_specified") is True
