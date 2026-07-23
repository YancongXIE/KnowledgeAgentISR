from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from neo4j import Driver
from pydantic import BaseModel, Field, ValidationError
from .analyzing_agent import AnalyzingAgent
from .elicitation_agent import ElicitationAgent
from .extraction_agent import ExtractionAgent
from .graph_query_agent import GraphQueryAgent
from .integration_agent import IntegrationAgent, render_knowledge_package_markdown
from .internalization_agent import InternalizationAgent
from .llm_json import complete_json
from .models import IntegrationState, RetrievalPlan
from .reflection_agent import ReflectionAgent
from .schema_agent import SchemaAgent
from .session_store import GLOBAL_SESSION_STORE, SessionStore
from .state import AgentState
from .summarizing_agent import SummarizingAgent
from .vicarious_learning_agent import VicariousLearningAgent

logger = logging.getLogger(__name__)

class _PdfChunkFocusJson(BaseModel):
    retrieval_instruction: str = Field(description='Natural-language instruction for the next PDF/chunk vector retrieval pass')
    rationale: str = Field(default='', description='Why this PDF focus helps answer the question')

class _AnalysisSelectionJson(BaseModel):
    selected_analyses: List[str] = Field(default_factory=list, description='Ordered candidate analysis function_name values; first item is the next step.')
    objective: str = Field(default='', description='Brief objective for the selected next analysis.')
    rationale: str = Field(default='', description='Why this next analysis is the best choice now.')

def _chunk_snippets(records: List[dict], *, max_chunks: int=4, max_chars: int=400) -> str:
    parts: List[str] = []
    for i, r in enumerate(records[:max_chunks]):
        text = r.get('text')
        if isinstance(text, str) and text.strip():
            t = text.strip()[:max_chars]
            parts.append(f'[chunk {i + 1}] {t}')
    return '\n'.join(parts) if parts else '(no chunk text yet)'

def _kg_snippets(records: List[dict], *, max_rows: int=6, max_chars: int=300) -> str:
    lines: List[str] = []
    for i, r in enumerate(records[:max_rows]):
        try:
            s = json.dumps(r, ensure_ascii=False)[:max_chars]
        except (TypeError, ValueError):
            s = str(r)[:max_chars]
        lines.append(f'[row {i + 1}] {s}')
    return '\n'.join(lines) if lines else '(no graph rows yet)'


class AzureOpenAIClient:
    """Wrapper around the Azure OpenAI SDK that exposes .chat(), .generate(), and .embeddings()
    with the same interface the orchestrator and agents expect."""

    def __init__(self, *, endpoint: str, api_key: str, api_version: str, deployment: str) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                'AzureOpenAIClient requires the `openai` package. '
                'Install with: pip install openai'
            ) from exc
        self._client = AzureOpenAI(
            azure_endpoint=endpoint.rstrip('/'),
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        format: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        opts = options or {}
        temperature = float(opts.get('temperature', 0.0))
        max_tokens = int(opts.get('num_predict', 1024))
        kwargs: Dict[str, Any] = {
            'model': self._deployment,
            'messages': messages,
            'temperature': temperature,
            'max_completion_tokens': max_tokens,
        }
        if format == 'json':
            kwargs['response_format'] = {'type': 'json_object'}
        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or '').strip()
        return {'message': {'content': text}}

    def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ]
        resp = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=2048,
        )
        text = (resp.choices[0].message.content or '').strip()
        return {'response': text}

    def embeddings(self, *, model: str, prompt: str) -> Dict[str, Any]:
        resp = self._client.embeddings.create(
            model=model or self._deployment,
            input=prompt,
        )
        emb = resp.data[0].embedding if resp.data else []
        return {'embedding': emb}

@dataclass
class OrchestratorResult:
    answer: str
    state: AgentState
    iterations_used: int
    status: str = 'complete'
    session_id: Optional[str] = None
    clarifying_questions: Optional[List[str]] = None
    collaboration_prompt: Optional[str] = None

class GraphState(TypedDict, total=False):
    question: str
    question_embedding: Optional[List[float]]
    scratchpad: List[str]
    available_actions: List[str]
    selected_actions: List[str]
    completed_actions: List[str]
    interpretation: Optional[Dict[str, Any]]
    retrieval_plan: Optional[Dict[str, Any]]
    last_cypher: Optional[str]
    last_records: List[Dict[str, Any]]
    extracted_records: List[Dict[str, Any]]
    last_error: Optional[str]
    analysis_report: Optional[Dict[str, Any]]
    analysis_note: Optional[str]
    evidence_summary: Optional[str]
    summary_cited_dois: List[str]
    integration_memo: str
    integration_trace: List[Dict[str, Any]]
    iteration: int
    extra: Optional[str]
    continue_loop: bool
    final_answer: Optional[str]
    integration_note: str
    pdf_focus_instruction: Optional[str]
    cycle_index: int
    selected_analyses: List[str]
    analysis_queue: List[str]
    analysis_step: int
    stop_gathering: bool
    current_objective: str
    selected_levels: List[str]
    use_kg: bool
    research_intent: Optional[Dict[str, Any]]
    integration_state: Optional[Dict[str, Any]]
    reflection: Optional[Dict[str, Any]]
    vicarious: Optional[Dict[str, Any]]
    knowledge_package: Optional[Dict[str, Any]]
    recommended_analyses: List[str]
    interactive: bool
    pause_status: Optional[str]
    resume_from: Optional[str]
    cycle_knowledge_packages: List[Dict[str, Any]]
    current_cycle_package: Optional[Dict[str, Any]]
    agent_memories: List[Dict[str, Any]]
    internalization_result: Optional[Dict[str, Any]]
    human_feedback: Optional[str]
    collaboration_prompt: Optional[str]
    await_human_gate: bool

class KGMultiAgentOrchestrator:
    ACTION_CATALOG: Dict[str, str] = {
        'elicit': 'Infer structured research intent from the human prompt',
        'plan_kg_from_context': 'Decide KG retrieval focus from current context (Extraction Pipeline)',
        'run_kg_query': 'Execute read-only Cypher against KG (Extraction Pipeline)',
        'run_analysis': 'Run analysis capabilities on KG query results (Extraction Pipeline)',
        'plan_pdf_from_kg': 'Decide next PDF chunk focus from KG results (Extraction Pipeline)',
        'pdf_refine': 'Vector-retrieve chunks using KG-informed instruction (Extraction Pipeline)',
        'build_cycle_knowledge_package': 'Assemble intermediate Knowledge Package for this extraction cycle',
        'integrate_state': 'Synthesize IntegrationState across salient evidence',
        'reflect': 'Decide whether another extraction cycle is worthwhile',
        'vicarious_learning': 'Retrieve qualitative readings for human internalization',
        'build_knowledge_package': 'Assemble the final Knowledge Package',
        'internalize': 'Persist co-constructed knowledge into AgentMemory (RAG internalization)',
        'summarize_answer': 'Lightweight evidence summary (baseline / package input)',
        'pause_exit': 'Exit graph while waiting for human clarification or collaboration feedback',
    }
    ANALYSIS_CAPABILITIES: Dict[str, List[Dict[str, str]]] = {'concept': [{'function_name': 'related_publications', 'description': 'This function identifies the publications that are related to the concept.'}, {'function_name': 'definitions', 'description': 'This function identifies the definitions of the concept.'}, {'function_name': 'definition_similarity', 'description': 'This function calculates the similarity score among concept definitions.'}, {'function_name': 'related_theories', 'description': 'This function identifies the theories that are related to the concept.'}], 'relationship': [{'function_name': 'antecedents_consequents', 'description': 'This function identifies the antecedents and consequents of a concept.'}, {'function_name': 'mediators_moderators', 'description': 'This function identifies the mediators and moderators of a relationship.'}, {'function_name': 'indegreecentrality', 'description': 'This function calculates the indegree centrality of a concept. Higher indegree centrality indicates more incoming connections, and popular consequents.'}, {'function_name': 'outdegreecentrality', 'description': 'This function calculates the outdegree centrality of a concept. Higher outdegree centrality indicates more outgoing connections, and fundamental antecedents.'}, {'function_name': 'betweennesscentrality', 'description': 'This function calculates the betweenness centrality of a concept. Higher betweenness centrality indicates influence as a bridge between other concepts.'}, {'function_name': 'cutpoints', 'description': 'This function identifies the cutpoints of a relationship. Cutpoints are concepts that, if removed, would disconnect the graph.'}, {'function_name': 'periphery', 'description': 'This function identifies the periphery index of a concept. Higher periphery index indicates concepts are close to the edge of the graph and more peripheral or innovative.'}, {'function_name': 'structural_hole_measures', 'description': 'This function calculates the structural hole measures of a concept. Constraint and effective size are calculated.'}, {'function_name': 'association_rules', 'description': 'This function identifies concepts tend to occur together in the same conceptual model.'}, {'function_name': 'knowledge_index', 'description': 'Knowledge index (KI) reflects conceptual convergence. Higher KI values mean antecedent paths are more convergent. A few antecedent paths dominate the explanation of focal dependent concept.Lower KI values mean antecedent paths are more divergent.'}]}
    _MEMO_ANCHOR_PREFIXES: tuple[str, ...] = ('[analysis_queue_seed]', '[select_analysis]', '[cycle_objective]', '[checkpoint]', '[elicitation]', '[reflection]')

    @classmethod
    def _function_name_to_level(cls) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for level, items in cls.ANALYSIS_CAPABILITIES.items():
            for item in items:
                fn = item.get('function_name')
                if fn:
                    out[fn] = level
        return out

    @classmethod
    def _allowed_analysis_function_names(cls) -> frozenset[str]:
        return frozenset(cls._function_name_to_level().keys())

    @staticmethod
    def _levels_from_analyses(analyses: List[str], fn_to_level: Dict[str, str]) -> List[str]:
        order = ('concept', 'relationship')
        seen: List[str] = []
        for a in analyses:
            lv = fn_to_level.get(a)
            if lv and lv not in seen:
                seen.append(lv)
        return [x for x in order if x in seen]

    def __init__(self, driver: Driver, llm_client: Any, embed_fn: Callable[[List[str]], List[List[float]]], *, model: str='gpt-5.2', max_iterations: int=4, max_cycles: int=4, default_candidate_pool: int=100, log_progress: bool=False, log_sink: Optional[Callable[[str], None]]=None, use_kg: bool=True, session_store: Optional[SessionStore]=None) -> None:
        self._driver = driver
        self._embed_fn = embed_fn
        self._max_iterations = max_iterations
        self._max_cycles = max_cycles
        self._candidate_pool = default_candidate_pool
        self._log_progress = log_progress
        self._log_sink = log_sink or (lambda msg: print(msg, flush=True))
        self._llm_client = llm_client
        self._model = model
        self._use_kg = bool(use_kg)
        self._session_store = session_store or GLOBAL_SESSION_STORE
        self.schema = SchemaAgent()
        self.graph_query = GraphQueryAgent(llm_client, self.schema, model=model)
        self.extraction = ExtractionAgent(embed_fn)
        _curation = os.environ.get('ANALYSIS_CURATION_ID', '13')
        self.analyzing = AnalyzingAgent(embed_fn, driver=driver, curation_id=str(_curation))
        self.summarizing = SummarizingAgent(llm_client, model=model)
        self.integration = IntegrationAgent(llm_client, model=model)
        self.elicitation = ElicitationAgent(llm_client, model=model)
        self.reflection_agent = ReflectionAgent(llm_client, model=model)
        self.vicarious = VicariousLearningAgent(llm_client, self.extraction, model=model)
        self.internalization = InternalizationAgent(driver)
        self._graph = self._build_graph()

    def _emit(self, message: str) -> None:
        if not self._log_progress:
            return
        ts = datetime.now().strftime('%H:%M:%S')
        self._log_sink(f'[{ts}] [orchestrator] {message}')

    def _compose_final_answer(self, state: AgentState, summary: str) -> str:
        if summary.strip():
            return summary.strip()
        if state.integration_memo.strip():
            return f'Question: {state.question}\nAvailable integrated evidence is limited. Best context so far:\n{state.integration_memo.strip()}'
        return f'Question: {state.question}\nInsufficient evidence gathered.'

    @staticmethod
    def _ensure_question_anchor(question: str, answer: str) -> str:
        q = (question or '').strip()
        a = (answer or '').strip()
        if not q or not a:
            return a
        first = a.splitlines()[0].strip().lower() if a.splitlines() else ''
        if first.startswith('question:') or first.startswith('answer:'):
            return a
        if q.lower() in a.lower():
            return a
        return f'Question: {q}\nAnswer: {a}'

    def _initial_graph_state(self, *, question: str, question_embedding: Optional[List[float]], use_kg: bool, interactive: bool = False) -> GraphState:
        memories: List[Dict[str, Any]] = []
        try:
            memories = self.internalization.retrieve(
                question_embedding=question_embedding,
                top_k=3,
                query_text=question,
            )
        except Exception as exc:
            logger.debug('AgentMemory retrieve at start failed: %s', exc)
        return {
            'question': question,
            'question_embedding': question_embedding,
            'scratchpad': [],
            'available_actions': list(self.ACTION_CATALOG.keys()),
            'selected_actions': [],
            'completed_actions': [],
            'interpretation': None,
            'retrieval_plan': None,
            'last_cypher': None,
            'last_records': [],
            'extracted_records': [],
            'last_error': None,
            'analysis_report': None,
            'analysis_note': None,
            'evidence_summary': None,
            'summary_cited_dois': [],
            'integration_memo': '',
            'integration_trace': [],
            'iteration': 0,
            'extra': None,
            'continue_loop': True,
            'final_answer': None,
            'integration_note': '',
            'pdf_focus_instruction': None,
            'cycle_index': 0,
            'selected_analyses': [],
            'analysis_queue': [],
            'analysis_step': 0,
            'stop_gathering': False,
            'use_kg': use_kg,
            'research_intent': None,
            'integration_state': None,
            'reflection': None,
            'vicarious': None,
            'knowledge_package': None,
            'recommended_analyses': [],
            'interactive': bool(interactive),
            'pause_status': None,
            'resume_from': None,
            'cycle_knowledge_packages': [],
            'current_cycle_package': None,
            'agent_memories': memories,
            'internalization_result': None,
            'human_feedback': None,
            'collaboration_prompt': None,
            'await_human_gate': False,
        }

    @staticmethod
    def _effective_question(state: GraphState) -> str:
        intent = state.get('research_intent') or {}
        refined = intent.get('refined_question') if isinstance(intent, dict) else None
        if isinstance(refined, str) and refined.strip():
            return refined.strip()
        return (state.get('question') or '').strip()

    def _run_single(self, *, question: str, question_embedding: Optional[List[float]], use_kg: bool, interactive: bool = False, initial_overrides: Optional[Dict[str, Any]] = None) -> OrchestratorResult:
        initial_state = self._initial_graph_state(
            question=question,
            question_embedding=question_embedding,
            use_kg=use_kg,
            interactive=interactive,
        )
        if initial_overrides:
            initial_state.update(initial_overrides)
        final_graph_state = self._graph.invoke(initial_state)
        return self._result_from_graph_state(final_graph_state, interactive=interactive, use_kg=use_kg)

    def _result_from_graph_state(
        self,
        final_graph_state: Dict[str, Any],
        *,
        interactive: bool,
        use_kg: bool,
        compare_baseline: bool = False,
        existing_session_id: Optional[str] = None,
    ) -> OrchestratorResult:
        final_state = self._agent_state_from_graph(final_graph_state)
        pause_status = final_graph_state.get('pause_status')
        if pause_status in ('needs_clarification', 'awaiting_human_feedback'):
            phase = 'elicitation' if pause_status == 'needs_clarification' else 'human_collaboration'
            intent = final_graph_state.get('research_intent') if isinstance(final_graph_state.get('research_intent'), dict) else None
            questions = list((intent or {}).get('clarifying_questions') or [])
            collab = final_graph_state.get('collaboration_prompt') or ''
            session = self._session_store.create(
                phase=phase,
                graph_state=dict(final_graph_state),
                research_intent=intent,
                clarifying_questions=questions,
                collaboration_prompt=collab,
                interactive=interactive,
                use_kg=use_kg,
                compare_baseline=compare_baseline,
                session_id=existing_session_id,
            )
            answer = collab if pause_status == 'awaiting_human_feedback' else self._format_clarification_answer(questions, intent)
            return OrchestratorResult(
                answer=answer,
                state=final_state,
                iterations_used=final_state.iteration,
                status=pause_status,
                session_id=session.session_id,
                clarifying_questions=questions or None,
                collaboration_prompt=collab or None,
            )
        final_answer = final_graph_state.get('final_answer') or self._compose_final_answer(final_state, final_state.evidence_summary or '')
        return OrchestratorResult(
            answer=final_answer,
            state=final_state,
            iterations_used=final_state.iteration,
            status='complete',
        )

    @staticmethod
    def _format_clarification_answer(questions: List[str], intent: Optional[Dict[str, Any]]) -> str:
        lines = [
            'Before I search the knowledge graph and literature, I need a clearer research focus.',
            '',
            'What this helps me do:',
            '- Retrieve the right concepts and evidence',
            '- Avoid a vague or off-target synthesis',
        ]
        refined = (intent or {}).get('refined_question') if isinstance(intent, dict) else None
        if refined:
            lines.extend(['', 'My current best guess of your question:', f'- {refined}'])
        lines.extend(['', 'Please reply with short answers to:'])
        if questions:
            for i, q in enumerate(questions, 1):
                lines.append(f'{i}. {q}')
        else:
            lines.append('1. What is your main research objective?')
            lines.append('2. Which focal concept(s) should I prioritize?')
            lines.append('3. What kind of output do you want (definition, relationships, proposition, gap)?')
        lines.extend(
            [
                '',
                'Tip: one short paragraph covering those points is enough.',
            ]
        )
        return '\n'.join(lines)

    @staticmethod
    def _plain_analysis_labels(names: List[str]) -> List[str]:
        labels = {
            'related_publications': 'related publications',
            'definitions': 'construct definitions',
            'definition_similarity': 'compare definitions across sources',
            'related_theories': 'related theories',
            'antecedents_consequents': 'antecedents and consequents',
            'mediators_moderators': 'mediators and moderators',
            'indegreecentrality': 'most common consequents (in-degree)',
            'outdegreecentrality': 'most common antecedents (out-degree)',
            'betweennesscentrality': 'bridge concepts (betweenness)',
            'cutpoints': 'cut-point concepts',
            'periphery': 'peripheral / innovative concepts',
            'structural_hole_measures': 'structural-hole measures',
            'association_rules': 'concepts that co-occur in models',
            'knowledge_index': 'knowledge-index / path convergence',
        }
        out: List[str] = []
        for name in names:
            if not isinstance(name, str) or not name.strip():
                continue
            key = name.strip()
            if key.lower().startswith('follow_up:'):
                continue
            out.append(labels.get(key, key.replace('_', ' ')))
        return out

    @staticmethod
    def _short_user_facing_text(text: str, *, max_chars: int = 260) -> str:
        cleaned = ' '.join((text or '').strip().split())
        if not cleaned:
            return ''
        # Prefer first 1–2 sentences for readability.
        parts = []
        buf = ''
        for ch in cleaned:
            buf += ch
            if ch in '.!?' and len(buf.strip()) >= 40:
                parts.append(buf.strip())
                buf = ''
                if len(parts) >= 2:
                    break
        summary = ' '.join(parts) if parts else cleaned
        if len(summary) <= max_chars:
            return summary
        cut = summary[: max_chars - 1].rsplit(' ', 1)[0]
        return (cut or summary[: max_chars - 1]).rstrip('.,;:') + '…'

    @classmethod
    def _format_collaboration_prompt(cls, *, reflection: Optional[Dict[str, Any]], integration_state: Optional[Dict[str, Any]]) -> str:
        refl = reflection if isinstance(reflection, dict) else {}
        integ = integration_state if isinstance(integration_state, dict) else {}

        lines = [
            'Checkpoint: I recommend gathering a bit more evidence before the final synthesis.',
            '',
            'You do not need to run any analyses yourself.',
            'Choose an action below, or type a short focus instruction.',
        ]

        why = cls._short_user_facing_text(str(refl.get('rationale') or ''))
        if why:
            lines.extend(['', 'Why I paused:', f'- {why}'])

        concepts = [str(c).strip() for c in (integ.get('emerging_concepts') or []) if str(c).strip()][:6]
        if concepts:
            lines.extend(['', 'Useful so far (emerging concepts):'])
            for c in concepts:
                lines.append(f'- {c}')

        gaps = [str(g).strip() for g in (integ.get('theoretical_gaps') or []) if str(g).strip()][:4]
        if gaps:
            lines.extend(
                [
                    '',
                    'Still missing from the literature evidence (not tasks for you):',
                ]
            )
            for g in gaps:
                lines.append(f'- {g}')

        next_analyses = cls._plain_analysis_labels(list(refl.get('recommended_analyses') or [])[:4])
        follow = (refl.get('follow_up_question') or '').strip()

        lines.extend(['', 'What I need from you (pick one):'])
        if next_analyses:
            lines.append(
                '1. Reply continue — I will search/analyze next for: '
                + '; '.join(next_analyses)
            )
        else:
            lines.append('1. Reply continue — I will run another evidence-gathering cycle')
        lines.append('2. Reply stop — skip more search and move to reading suggestions + final package')
        if follow:
            short_follow = cls._short_user_facing_text(follow, max_chars=320)
            lines.append(f'3. Or type a focus note answering: {short_follow}')
        else:
            lines.append(
                '3. Or type a focus note (e.g., which concept, domain, or relationship to prioritize)'
            )

        lines.extend(
            [
                '',
                'Quick guide:',
                '- continue = keep searching',
                '- stop = finish now',
                '- free text = steer the next search (I still do the analyses)',
            ]
        )
        return '\n'.join(lines)

    @staticmethod
    def _append_log(state: GraphState, line: str) -> List[str]:
        scratch = list(state.get('scratchpad', []))
        scratch.append(line)
        return scratch

    @staticmethod
    def _merge_chunk_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        out: List[Dict[str, Any]] = []
        for row in existing + new_rows:
            key = (row.get('chunk_uuid'), row.get('doi'), (row.get('text') or '')[:120] if isinstance(row.get('text'), str) else None)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _agent_state_from_graph(self, state: GraphState) -> AgentState:
        mapped: Dict[str, Any] = {'question': state['question']}
        for f in fields(AgentState):
            key = f.name
            if key == 'question' or key not in state:
                continue
            value = state.get(key)
            if isinstance(value, list):
                mapped[key] = list(value)
            elif isinstance(value, dict):
                mapped[key] = dict(value)
            else:
                mapped[key] = value
        return AgentState(**mapped)

    def _bump_iteration(self, state: GraphState) -> int:
        return state.get('iteration', 0) + 1

    @classmethod
    def _memo_window(cls, memo: str, *, max_chars: int) -> str:
        text = (memo or '').strip()
        if not text:
            return ''
        if len(text) <= max_chars:
            return text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return text[:max_chars]
        anchors: List[str] = []
        seen: set[str] = set()
        for ln in lines:
            if ln in seen:
                continue
            if ln.startswith(cls._MEMO_ANCHOR_PREFIXES):
                anchors.append(ln)
                seen.add(ln)
            if len(anchors) >= 6:
                break
        marker = '... [memo middle omitted] ...'
        selected_tail: List[str] = []
        used = len(marker) + 2
        for ln in reversed(lines):
            if ln in seen:
                continue
            add = len(ln) + 1
            if used + add > max_chars:
                break
            selected_tail.append(ln)
            used += add
            seen.add(ln)
        selected_tail.reverse()
        parts: List[str] = []
        if anchors:
            parts.extend(anchors)
        if anchors and selected_tail:
            parts.append(marker)
        parts.extend(selected_tail)
        out = '\n'.join(parts).strip()
        if len(out) <= max_chars:
            return out
        half = max(40, max_chars // 2)
        return f'{out[:half]}\n{marker}\n{out[-half:]}'

    def _fallback_analysis_for_cycle(self, cycle_index: int, allowed: frozenset[str], completed: set[str]) -> str:
        """Deterministic fallback when LLM selection fails."""
        if cycle_index <= 0:
            ordered = ['related_publications', 'definitions', 'related_theories']
        elif cycle_index == 1:
            ordered = ['antecedents_consequents', 'mediators_moderators', 'indegreecentrality', 'outdegreecentrality', 'betweennesscentrality']
        else:
            ordered = ['association_rules', 'knowledge_index', 'cutpoints', 'periphery', 'structural_hole_measures']
        for cand in ordered:
            if cand in allowed and cand not in completed:
                return cand
        for fallback in ('related_publications', 'definitions', 'antecedents_consequents'):
            if fallback in allowed and fallback not in completed:
                return fallback
        remaining = [c for c in allowed if c not in completed]
        if remaining:
            return remaining[0]
        return next(iter(allowed)) if allowed else 'related_publications'

    def _select_next_analysis(self, state: GraphState) -> str:
        allowed = self._allowed_analysis_function_names()
        already_run = [x for x in state.get('analysis_queue') or [] if x in allowed]
        completed = set(already_run)
        cycle_index = int(state.get('analysis_step', 0))
        for cand in state.get('recommended_analyses') or []:
            if isinstance(cand, str) and cand in allowed and cand not in completed:
                return cand
        system = 'Select the SINGLE best next Analysaurus analysis function (by function_name). The catalog in analysis_capabilities lists each function_name and description. Choose functions by interpreting the MEANING and PURPOSE of each analysis description, not by lexical overlap with the question text. Use research_intent, current state, prior analyses, and evidence summaries to decide what is most useful now. Avoid repeating analyses that were already run unless there is a strong reason from the evidence. Return JSON only with selected_analyses (first item is the next analysis), objective, rationale.'
        user = json.dumps({
            'question': self._effective_question(state),
            'research_intent': state.get('research_intent') or {},
            'cycle_index': cycle_index,
            'already_run_analyses': already_run,
            'recommended_by_reflection': list(state.get('recommended_analyses') or []),
            'analysis_capabilities': str(self.ANALYSIS_CAPABILITIES),
            'analysis_brief': self._analysis_brief(state.get('analysis_report')),
            'kg_rows_excerpt': _kg_snippets(list(state.get('last_records', []))),
            'pdf_chunks_excerpt': _chunk_snippets(list(state.get('extracted_records', []))),
            'context_memo': self._memo_window(state.get('integration_memo') or '', max_chars=2500),
        }, ensure_ascii=False, indent=2)
        try:
            out = complete_json(self._llm_client, model=self._model, system=system, user=user, schema_model=_AnalysisSelectionJson)
            raw = [x.strip() for x in out.selected_analyses if isinstance(x, str) and x.strip()]
            for cand in raw:
                if cand in allowed and cand not in completed:
                    return cand
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('Analysis selection LLM returned invalid output: %s', exc)
        except Exception as exc:
            logger.warning('Analysis selection LLM failed: %s: %s', type(exc).__name__, exc)
        return self._fallback_analysis_for_cycle(cycle_index, allowed, completed)

    def _plan_kg(self, *, question: str, found_records: List[dict], integration_memo: str, selected_analyses: List[str], selected_levels: List[str], objective: str, research_intent: Optional[Dict[str, Any]] = None, agent_memories: Optional[List[Dict[str, Any]]] = None) -> RetrievalPlan:
        system = 'You plan what to retrieve from a Neo4j knowledge graph (concepts, relations, publications) based on the structured research intent and evidence found so far. Output JSON matching the required keys. Do not write Cypher.\n\n' + self.schema.prompt_fragment()
        user = json.dumps({
            'user_question': question,
            'research_intent': research_intent or {},
            'evidence_excerpts': _chunk_snippets(found_records),
            'selected_analyses': selected_analyses,
            'selected_levels': selected_levels,
            'cycle_objective': objective,
            'running_context_memo': self._memo_window(integration_memo, max_chars=2200),
            'internalized_agent_memory': InternalizationAgent.format_for_prompt(list(agent_memories or [])),
        }, ensure_ascii=False, indent=2)
        try:
            return complete_json(self._llm_client, model=self._model, system=system, user=user, schema_model=RetrievalPlan)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('KG plan LLM returned invalid schema: %s', exc)
        except Exception as exc:
            logger.warning('KG plan LLM failed: %s: %s', type(exc).__name__, exc)
        q = question.lower()
        concepts = []
        if isinstance(research_intent, dict):
            concepts = list(research_intent.get('target_concepts') or [])
        rels = ['IS_ANTECEDENT_OF', 'IS_CONSEQUENT_OF'] if ('trust' in q or any('trust' in str(c).lower() for c in concepts)) else []
        return RetrievalPlan(intent='Iterative fallback: explore structure and publication-linked chunks.', target_labels=['Publication', 'Chunk', 'Element'], key_properties={}, preferred_relationships=rels, search_hint='Use graph patterns and vector search over chunks as needed.')

    def _plan_pdf(self, *, question: str, found_records: List[dict], integration_memo: str, research_intent: Optional[Dict[str, Any]] = None) -> str:
        system = 'You decide what to search next in PDF text chunks (vector retrieval) based on the research intent and evidence found so far (concepts, relations, definitions, DOIs, and other retrieved context). Return a focused natural-language retrieval_instruction for finding the most useful passages.'
        user = json.dumps({'user_question': question, 'research_intent': research_intent or {}, 'found_evidence_sample': _kg_snippets(found_records), 'running_context_memo': self._memo_window(integration_memo, max_chars=2200)}, ensure_ascii=False, indent=2)
        try:
            out = complete_json(self._llm_client, model=self._model, system=system, user=user, schema_model=_PdfChunkFocusJson)
            return out.retrieval_instruction.strip()
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('PDF plan LLM returned invalid schema: %s', exc)
        except Exception as exc:
            logger.warning('PDF plan LLM failed: %s: %s', type(exc).__name__, exc)
        return f'Find passages that help answer: {question}. Prioritize definitions, propositions, and empirical findings related to prior graph results.'

    def _cycle_objective(self, state: GraphState) -> str:
        analyses = list(state.get('selected_analyses', []))
        analysis_focus = ', '.join(analyses) if analyses else 'related_publications, definitions'
        cycle = int(state.get('analysis_step', 0))
        if cycle == 0:
            return f'Establish foundational evidence with analyses: {analysis_focus}.'
        if cycle == 1:
            return f'Deepen evidence and clarify constructs with analyses: {analysis_focus}.'
        return f'Close evidence gaps and support answer quality using analyses: {analysis_focus}.'

    @staticmethod
    def _enrich_plan_with_objective(plan: RetrievalPlan, objective: str, cycle: int, selected_levels: List[str]) -> RetrievalPlan:
        labels = list(dict.fromkeys(list(plan.target_labels) + ['Publication', 'Chunk', 'Element']))
        rels = list(plan.preferred_relationships)
        hint = (plan.search_hint or '').strip()
        if cycle == 0:
            labels = list(dict.fromkeys(labels + ['Definition', 'Theory']))
            rels = list(dict.fromkeys(rels + ['DEPICTS', 'HAS', 'HAS_CHUNK']))
        elif cycle == 1:
            labels = list(dict.fromkeys(labels + ['Definition', 'Theory']))
            rels = list(dict.fromkeys(rels + ['HAS', 'DEPICTS']))
        else:
            labels = list(dict.fromkeys(labels + ['Relation', 'Model']))
            rels = list(dict.fromkeys(rels + ['HAS', 'DEPICTS']))
        if 'relationship' in selected_levels:
            labels = list(dict.fromkeys(labels + ['Relation', 'Model']))
        combined_hint = f'{objective} {hint}'.strip()
        return plan.model_copy(update={'target_labels': labels, 'preferred_relationships': rels, 'search_hint': combined_hint})

    @staticmethod
    def _analysis_brief(report: Dict[str, Any] | None) -> str:
        if not report:
            return ''
        pieces: List[str] = []
        l1 = report.get('level_1_concept_extraction', {})
        if isinstance(l1, dict):
            pieces.append(f'L1 pubs={l1.get('n_related_publications', 0)} defs={l1.get('definitions_found', 0)} theories={l1.get('n_related_theories', 0)}')
        l2 = report.get('level_2_relationship_mining', {})
        if isinstance(l2, dict):
            ind = l2.get('indegreecentrality') if isinstance(l2.get('indegreecentrality'), dict) else {}
            n_nodes = ind.get('n_nodes')
            n_edges = ind.get('n_edges')
            ns = str(n_nodes) if n_nodes is not None else '?'
            es = str(n_edges) if n_edges is not None else '?'
            ar = l2.get('association_rules') if isinstance(l2.get('association_rules'), dict) else {}
            n_rules = ar.get('n_rules')
            rs = str(n_rules) if isinstance(n_rules, int) else '0'
            ki = l2.get('knowledge_index')
            ki_dist = ki.get('consequent_distribution_ki') if isinstance(ki, dict) else None
            ki_s = f'{ki_dist:.3f}' if isinstance(ki_dist, (int, float)) else 'n/a'
            pieces.append(f'L2 nodes={ns} edges={es} rules={rs} KI_dist={ki_s}')
        return ' | '.join(pieces)

    @staticmethod
    def _filter_analysis_report(report: Dict[str, Any] | None, selected_levels: List[str]) -> Dict[str, Any] | None:
        if not report:
            return None
        if not selected_levels:
            return report
        mapping = {'concept': 'level_1_concept_extraction', 'relationship': 'level_2_relationship_mining'}
        keys = {mapping[level] for level in selected_levels if level in mapping}
        filtered = {k: v for k, v in report.items() if k in keys}
        return filtered or report

    def _node_elicit(self, state: GraphState) -> GraphState:
        self._emit('node=elicit start')
        intent = self.elicitation.elicit(state.get('question', ''))
        payload = intent.model_dump()
        scratch = self._append_log(
            state,
            f"[elicitation] specified={intent.is_sufficiently_specified} discovery_type={intent.discovery_type!r} concepts={intent.target_concepts[:6]}",
        )
        if intent.clarifying_questions:
            scratch.append(f'[elicitation] clarifying_questions={intent.clarifying_questions[:4]}')
        memories = list(state.get('agent_memories') or [])
        if memories:
            scratch.append(f'[elicitation] agent_memories={len(memories)}')
        interactive = bool(state.get('interactive'))
        needs_pause = bool(interactive and intent.is_sufficiently_specified is False)
        out: GraphState = {
            'research_intent': payload,
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
            'pause_status': 'needs_clarification' if needs_pause else None,
        }
        if needs_pause:
            out['final_answer'] = self._format_clarification_answer(list(intent.clarifying_questions or []), payload)
            scratch.append('[elicitation] pause=needs_clarification')
            out['scratchpad'] = scratch
        self._emit('node=elicit done')
        return out

    def _route_after_elicit(self, state: GraphState) -> str:
        if state.get('pause_status') == 'needs_clarification':
            self._emit('route elicit -> pause_exit (needs_clarification)')
            return 'pause_exit'
        self._emit('route elicit -> interpret')
        return 'interpret'

    def _node_pause_exit(self, state: GraphState) -> GraphState:
        self._emit(f"node=pause_exit status={state.get('pause_status')}")
        return {}

    def _node_interpret(self, state: GraphState) -> GraphState:
        self._emit('node=interpret start')
        intent = state.get('research_intent') or {}
        interpretation = {
            'strategy': 'adaptive',
            'note': 'KDE flow: elicitation -> extraction pipeline -> integration/reflection -> vicarious learning.',
            'discovery_type': intent.get('discovery_type') if isinstance(intent, dict) else None,
            'refined_question': intent.get('refined_question') if isinstance(intent, dict) else None,
        }
        self._emit('node=interpret done strategy=adaptive')
        scratch = self._append_log(state, '[interpret] strategy=adaptive kde=true')
        scratch.append('[analysis_queue] empty at start; selector chooses next analysis each cycle')
        return {'available_actions': list(self.ACTION_CATALOG.keys()), 'interpretation': interpretation, 'analysis_queue': [], 'analysis_step': 0, 'stop_gathering': False, 'selected_analyses': [], 'integration_memo': state.get('integration_memo', ''), 'research_intent': state.get('research_intent'), 'scratchpad': scratch, 'iteration': self._bump_iteration(state), 'use_kg': bool(state.get('use_kg', self._use_kg))}

    def _route_after_interpret(self, state: GraphState) -> str:
        if bool(state.get('use_kg', self._use_kg)):
            self._emit('route interpret -> select_analysis (kg enabled)')
            return 'select_analysis'
        self._emit('route interpret -> baseline_retrieve (kg disabled)')
        return 'baseline_retrieve'

    def _node_baseline_retrieve(self, state: GraphState) -> GraphState:
        self._emit('node=baseline_retrieve start')
        question = state['question']
        rows, note = self.extraction.extract_from_instruction(self._driver, question, question=question, question_embedding=state.get('question_embedding'), fallback_records=None)
        merged = self._merge_chunk_rows(list(state.get('extracted_records', [])), rows)
        scratch = self._append_log(state, f'[baseline_retrieve] {note}')
        self._emit(f'node=baseline_retrieve done total_chunks={len(merged)}')
        return {'extracted_records': merged, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state), 'stop_gathering': True}

    def _node_plan_kg_from_context(self, state: GraphState) -> GraphState:
        cycle = int(state.get('analysis_step', 0))
        self._emit('node=plan_kg_from_context start analysis_step={}'.format(cycle))
        analyses = list(state.get('selected_analyses', []))
        if not analyses:
            analyses = [self._select_next_analysis(state)]
        fn_to_level = self._function_name_to_level()
        levels = list(state.get('selected_levels', []))
        if not levels:
            levels = self._levels_from_analyses(analyses, fn_to_level)
        if not levels:
            levels = ['concept', 'relationship']
        objective = (state.get('current_objective') or '').strip() or self._cycle_objective({**state, 'selected_analyses': analyses})
        plan = self._plan_kg(
            question=self._effective_question(state),
            found_records=list(state.get('extracted_records', [])),
            integration_memo=state.get('integration_memo', ''),
            selected_analyses=analyses,
            selected_levels=levels,
            objective=objective,
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            agent_memories=list(state.get('agent_memories') or []),
        )
        plan = self._enrich_plan_with_objective(plan, objective, cycle, levels)
        scratch = self._append_log(state, f'[extraction_pipeline][plan_kg_from_context] cycle={cycle} objective={objective}')
        scratch.append(f'[plan_kg_from_context] selected_analyses={analyses} derived_levels={levels}')
        scratch.append(f'[plan_kg_from_context] intent={plan.intent!r}')
        self._emit('node=plan_kg_from_context done')
        return {'retrieval_plan': plan.model_dump(), 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_select_analysis(self, state: GraphState) -> GraphState:
        cycle = int(state.get('analysis_step', 0))
        self._emit('node=select_analysis start analysis_step={}'.format(cycle))
        next_analysis = self._select_next_analysis(state)
        queue = list(state.get('analysis_queue') or [])
        queue.append(next_analysis)
        analyses = [next_analysis]
        fn_to_level = self._function_name_to_level()
        levels = self._levels_from_analyses(analyses, fn_to_level)
        if not levels:
            levels = ['concept', 'relationship']
        objective = self._cycle_objective({**state, 'selected_analyses': analyses})
        scratch = self._append_log(state, f'[select_analysis] cycle={cycle} selected={analyses}')
        scratch.append(f'[select_analysis] levels={levels} objective={objective}')
        self._emit('node=select_analysis done')
        return {'analysis_queue': queue, 'selected_analyses': analyses, 'selected_levels': levels, 'current_objective': objective, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_run_kg_query(self, state: GraphState) -> GraphState:
        self._emit('node=run_kg_query start')
        question = self._effective_question(state)
        question_embedding = state.get('question_embedding')
        rp = state.get('retrieval_plan') or {}
        plan = RetrievalPlan.model_validate(rp)
        intent_bit = ''
        intent = state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None
        if intent:
            intent_bit = f" research_intent={json.dumps({k: intent.get(k) for k in ('objective', 'target_concepts', 'discovery_type')}, ensure_ascii=False)[:600]}"
        bundle = self.graph_query.propose_query(question, plan, question_embedding=question_embedding, extra_context=self._memo_window(state.get('integration_memo', ''), max_chars=1800) + intent_bit)
        merged = dict(bundle.parameters)
        if question_embedding is not None:
            if 'embedding' not in merged and '$embedding' in bundle.cypher:
                merged['embedding'] = question_embedding
            if 'candidate_pool' not in merged and 'queryNodes' in bundle.cypher:
                merged['candidate_pool'] = self._candidate_pool
        bundle_exec = bundle.model_copy(update={'parameters': merged})
        records, err = self.graph_query.execute_safe(self._driver, bundle_exec)
        scratch = self._append_log(state, f'[kg_query] rows={len(records)} err={err}')
        self._emit(f'node=run_kg_query done rows={len(records)} err={err}')
        return {'last_cypher': bundle.cypher, 'last_records': records, 'last_error': err, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_run_analysis(self, state: GraphState) -> GraphState:
        self._emit('node=run_analysis start')
        records = list(state.get('last_records', []))
        analysis_report: Dict[str, Any] | None = None
        analysis_note = ''
        analysis_line = ''
        if records:
            selected = list(state.get('selected_analyses', []))
            analysis_report, analysis_note = self.analyzing.analyze_selected(records, selected_analyses=selected)
            analysis_line = self._analysis_brief(analysis_report)
        memo = state.get('integration_memo', '')
        scratch = list(state.get('scratchpad', []))
        if analysis_line:
            scratch.append(f'[analysis_brief] {analysis_line}')
        if analysis_note:
            scratch.append(f'[analysis] {analysis_note}')
        self._emit(f'node=run_analysis done rows={len(records)}')
        return {'analysis_report': analysis_report, 'analysis_note': analysis_note, 'integration_memo': memo, 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_plan_pdf_from_kg(self, state: GraphState) -> GraphState:
        self._emit('node=plan_pdf_from_kg start')
        instruction = self._plan_pdf(
            question=self._effective_question(state),
            found_records=list(state.get('last_records', [])),
            integration_memo=state.get('integration_memo', ''),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
        )
        scratch = self._append_log(state, f'[extraction_pipeline][plan_pdf_from_kg] instruction={instruction[:120]!r}...')
        self._emit('node=plan_pdf_from_kg done')
        return {'pdf_focus_instruction': instruction, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_pdf_refine(self, state: GraphState) -> GraphState:
        self._emit('node=pdf_refine start')
        question = self._effective_question(state)
        step = int(state.get('analysis_step', 0))
        instr = (state.get('pdf_focus_instruction') or question).strip()
        new_rows, note = self.extraction.extract_from_instruction(self._driver, instr, question=question, question_embedding=state.get('question_embedding'), fallback_records=list(state.get('last_records', [])))
        merged = self._merge_chunk_rows(list(state.get('extracted_records', [])), new_rows)
        scratch = self._append_log(state, f'[extraction_pipeline][pdf_refine] analysis_step={step} {note}')
        self._emit(f'node=pdf_refine done analysis_step={step} total_chunks={len(merged)}')
        return {'extracted_records': merged, 'cycle_index': step, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_build_cycle_knowledge_package(self, state: GraphState) -> GraphState:
        self._emit('node=build_cycle_knowledge_package start')
        cycle = int(state.get('analysis_step', 0))
        chunk_summary = _chunk_snippets(list(state.get('extracted_records', [])), max_chunks=6, max_chars=500)
        package = self.integration.build_cycle_package_from_extraction(
            question=self._effective_question(state),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            kg_records=list(state.get('last_records', [])),
            analysis_report=state.get('analysis_report') if isinstance(state.get('analysis_report'), dict) else None,
            analysis_note=state.get('analysis_note') or '',
            chunk_summary=chunk_summary,
            cycle_index=cycle,
            last_cypher=state.get('last_cypher'),
        )
        dump = package.model_dump()
        packages = list(state.get('cycle_knowledge_packages') or [])
        packages.append(dump)
        scratch = self._append_log(
            state,
            f'[cycle_knowledge_package] stage={package.stage} concepts={len(package.key_concepts)} evidence={len(package.supporting_evidence)}',
        )
        self._emit('node=build_cycle_knowledge_package done')
        return {
            'current_cycle_package': dump,
            'cycle_knowledge_packages': packages,
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
        }

    def _node_integrate_state(self, state: GraphState) -> GraphState:
        self._emit('node=integrate_state start')
        prior = state.get('integration_state')
        out = self.integration.integrate_knowledge(
            stage=f'cycle_{int(state.get("analysis_step", 0))}',
            question=self._effective_question(state),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            pdf_summary=_chunk_snippets(list(state.get('extracted_records', [])), max_chunks=8, max_chars=600),
            prior_state=prior if isinstance(prior, dict) else None,
            kg_records=list(state.get('last_records', [])),
            analysis_report=state.get('analysis_report'),
            analysis_note=state.get('analysis_note') or '',
            cycle_package=state.get('current_cycle_package') if isinstance(state.get('current_cycle_package'), dict) else None,
        )
        merged_context = (out.merged_context or '').strip()
        memo = merged_context or state.get('integration_memo', '')
        integ_dump = out.model_dump()
        trace = list(state.get('integration_trace', []))
        trace.append({
            'stage': out.stage or 'integrate_state',
            'analysis_step': int(state.get('analysis_step', 0)),
            'key_points': list(out.key_points or [])[:8],
            'next_focus': (out.next_focus or '')[:300],
            'emerging_concepts': list(out.emerging_concepts or [])[:8],
            'confidence': out.confidence,
        })
        scratch = self._append_log(
            state,
            f'[integrate_state] concepts={len(out.emerging_concepts or [])} props={len(out.propositions or [])} confidence={out.confidence:.2f}',
        )
        self._emit('node=integrate_state done')
        return {
            'integration_state': integ_dump,
            'integration_memo': memo,
            'integration_trace': trace,
            'integration_note': out.next_focus or '',
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
        }

    def _node_checkpoint(self, state: GraphState) -> GraphState:
        """Reflection step: decide whether another Extraction Pipeline cycle is worthwhile."""
        self._emit('node=checkpoint/reflection start')
        step = int(state.get('analysis_step', 0))
        next_step = step + 1
        allowed = sorted(self._allowed_analysis_function_names())
        already_run = [x for x in (state.get('analysis_queue') or []) if x in self._allowed_analysis_function_names()]
        decision = self.reflection_agent.reflect(
            question=self._effective_question(state),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            integration_state=state.get('integration_state') if isinstance(state.get('integration_state'), dict) else None,
            analysis_brief=self._analysis_brief(state.get('analysis_report')),
            already_run_analyses=already_run,
            allowed_analyses=allowed,
            cycle_index=step,
            max_cycles=self._max_cycles,
            kg_excerpt=_kg_snippets(list(state.get('last_records', [])), max_rows=4, max_chars=350),
            chunk_excerpt=_chunk_snippets(list(state.get('extracted_records', [])), max_chunks=5, max_chars=400),
        )
        should_stop = not bool(decision.continue_loop)
        reflection_dump = decision.model_dump()
        interactive = bool(state.get('interactive'))
        await_human = bool(interactive and not should_stop)
        collaboration_prompt = ''
        if await_human:
            collaboration_prompt = self._format_collaboration_prompt(
                reflection=reflection_dump,
                integration_state=state.get('integration_state') if isinstance(state.get('integration_state'), dict) else None,
            )
            self._emit('node=checkpoint/reflection done pause=awaiting_human_feedback')
        elif should_stop:
            self._emit('node=checkpoint/reflection done stop -> vicarious_learning')
        else:
            self._emit(f'node=checkpoint/reflection done continue -> analysis_step={next_step}')
        trace = list(state.get('integration_trace', []))
        trace.append({
            'stage': 'reflection',
            'analysis_step': step,
            'sufficient': decision.sufficient,
            'continue_loop': decision.continue_loop,
            'recommended_analyses': list(decision.recommended_analyses or [])[:6],
            'rationale': (decision.rationale or '')[:500],
            'await_human_gate': await_human,
        })
        scratch = self._append_log(
            state,
            f'[reflection] sufficient={decision.sufficient} continue={decision.continue_loop} next={list(decision.recommended_analyses or [])[:4]} human_gate={await_human}',
        )
        out_state: GraphState = {
            'stop_gathering': should_stop,
            'reflection': reflection_dump,
            'recommended_analyses': list(decision.recommended_analyses or []),
            'integration_memo': state.get('integration_memo', ''),
            'scratchpad': scratch,
            'integration_trace': trace,
            'iteration': self._bump_iteration(state),
            'await_human_gate': await_human,
            'pause_status': 'awaiting_human_feedback' if await_human else None,
            'collaboration_prompt': collaboration_prompt or None,
            'final_answer': collaboration_prompt if await_human else state.get('final_answer'),
        }
        if not should_stop and not await_human:
            out_state['analysis_step'] = next_step
        elif not should_stop and await_human:
            # analysis_step advances on human continue
            out_state['analysis_step'] = step
        return out_state

    def _route_after_checkpoint(self, state: GraphState) -> str:
        if state.get('pause_status') == 'awaiting_human_feedback' or state.get('await_human_gate'):
            self._emit('route reflection -> pause_exit (awaiting_human_feedback)')
            return 'pause_exit'
        if state.get('stop_gathering'):
            self._emit('route reflection -> vicarious_learning')
            return 'vicarious_learning'
        self._emit('route reflection -> select_analysis')
        return 'select_analysis'

    def _prepare_analysis_for_summary(self, state: GraphState) -> tuple[Dict[str, Any] | None, str]:
        """Ensure we have an analysis report for the summarize step."""
        report = state.get('analysis_report')
        anote = state.get('analysis_note') or ''
        if report is None and state.get('last_records'):
            report, anote = self.analyzing.analyze(list(state.get('last_records', [])))
        return report, anote

    def _node_summarize_answer(self, state: GraphState) -> GraphState:
        """Baseline / lite summary path: summarize chunks and build a simplified IntegrationState."""
        self._emit('node=summarize_answer start')
        report, anote = self._prepare_analysis_for_summary(state)
        question = self._effective_question(state)
        summary_out = self.summarizing.summarize(list(state.get('extracted_records', [])), question=question)
        integ = self.integration.integrate_knowledge(
            stage='baseline_or_lite_summary',
            question=question,
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            pdf_summary=summary_out.summary,
            prior_state=state.get('integration_state') if isinstance(state.get('integration_state'), dict) else None,
            kg_records=list(state.get('last_records', [])),
            analysis_report=report,
            analysis_note=anote,
        )
        memo = (integ.merged_context or '').strip() or state.get('integration_memo', '')
        scratch = self._append_log(state, f'[summarize] len={len(summary_out.summary)} dois={len(summary_out.cited_dois)}')
        trace = list(state.get('integration_trace', []))
        trace.append({
            'stage': integ.stage or 'baseline_or_lite_summary',
            'analysis_step': int(state.get('analysis_step', 0)),
            'key_points': list(integ.key_points or [])[:8],
            'confidence': integ.confidence,
        })
        self._emit('node=summarize_answer done')
        return {
            'analysis_report': report,
            'analysis_note': anote,
            'evidence_summary': summary_out.summary,
            'summary_cited_dois': list(summary_out.cited_dois),
            'integration_state': integ.model_dump(),
            'integration_memo': memo,
            'integration_trace': trace,
            'integration_note': integ.next_focus or '',
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
        }

    def _node_vicarious_learning(self, state: GraphState) -> GraphState:
        self._emit('node=vicarious_learning start')
        # Ensure we have an evidence summary before packaging.
        report = state.get('analysis_report')
        anote = state.get('analysis_note') or ''
        evidence_summary = state.get('evidence_summary') or ''
        cited = list(state.get('summary_cited_dois') or [])
        if not evidence_summary.strip():
            report, anote = self._prepare_analysis_for_summary(state)
            summary_out = self.summarizing.summarize(
                list(state.get('extracted_records', [])),
                question=self._effective_question(state),
            )
            evidence_summary = summary_out.summary
            cited = list(summary_out.cited_dois)
        integ_state = state.get('integration_state')
        if not isinstance(integ_state, dict):
            integ = self.integration.integrate_knowledge(
                stage='pre_vicarious',
                question=self._effective_question(state),
                research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
                pdf_summary=evidence_summary,
                prior_state=None,
                kg_records=list(state.get('last_records', [])),
                analysis_report=report,
                analysis_note=anote,
            )
            integ_state = integ.model_dump()
        vic = self.vicarious.learn(
            self._driver,
            question=self._effective_question(state),
            question_embedding=state.get('question_embedding'),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            integration_state=integ_state,
            fallback_records=list(state.get('extracted_records', [])),
        )
        scratch = self._append_log(
            state,
            f'[vicarious_learning] readings={len(vic.reading_sequence)} studies={len(vic.illustrative_studies)} note={vic.note[:80]!r}',
        )
        self._emit('node=vicarious_learning done')
        return {
            'vicarious': vic.model_dump(),
            'analysis_report': report,
            'analysis_note': anote,
            'evidence_summary': evidence_summary,
            'summary_cited_dois': cited,
            'integration_state': integ_state,
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
        }

    def _node_build_knowledge_package(self, state: GraphState) -> GraphState:
        self._emit('node=build_knowledge_package start')
        integ = state.get('integration_state')
        if not isinstance(integ, dict):
            # Baseline path may skip vicarious but still need a package.
            lite = IntegrationState(
                stage='package_fallback',
                merged_context=state.get('integration_memo') or state.get('evidence_summary') or '',
                key_points=[],
                confidence=0.2,
            )
            integ = lite.model_dump()
        package = self.integration.build_knowledge_package(
            question=self._effective_question(state) or state.get('question', ''),
            research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
            integration_state=integ,
            evidence_summary=state.get('evidence_summary') or '',
            cited_dois=list(state.get('summary_cited_dois') or []),
            vicarious=state.get('vicarious') if isinstance(state.get('vicarious'), dict) else None,
            reflection=state.get('reflection') if isinstance(state.get('reflection'), dict) else None,
            last_cypher=state.get('last_cypher'),
            analysis_note=state.get('analysis_note') or '',
            stage='final',
        )
        markdown = render_knowledge_package_markdown(package)
        question = state.get('question', '')
        final = self._ensure_question_anchor(question, markdown)
        internalization_result: Dict[str, Any] | None = None
        try:
            embedding = state.get('question_embedding')
            if embedding is None and question:
                embedding = self._embed_fn([question])[0]
            internalization_result = self.internalization.persist(
                package=package,
                research_intent=state.get('research_intent') if isinstance(state.get('research_intent'), dict) else None,
                embedding=list(embedding) if embedding else None,
                question=question,
            )
        except Exception as exc:
            internalization_result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        scratch = self._append_log(
            state,
            f'[knowledge_package] concepts={len(package.key_concepts)} props={len(package.candidate_propositions)} confidence={package.confidence:.2f}',
        )
        if internalization_result:
            scratch.append(f'[internalization] {internalization_result}')
        self._emit('node=build_knowledge_package done')
        return {
            'knowledge_package': package.model_dump(),
            'final_answer': final,
            'integration_state': integ,
            'internalization_result': internalization_result,
            'scratchpad': scratch,
            'iteration': self._bump_iteration(state),
            'pause_status': None,
            'await_human_gate': False,
        }

    def _node_finalize(self, state: GraphState) -> GraphState:
        self._emit('node=finalize')
        if state.get('final_answer'):
            return {}
        pkg = state.get('knowledge_package')
        if isinstance(pkg, dict):
            try:
                from .models import KnowledgePackage
                return {'final_answer': render_knowledge_package_markdown(KnowledgePackage.model_validate(pkg))}
            except Exception:
                pass
        fallback = state.get('evidence_summary') or 'No sufficient evidence gathered.'
        return {'final_answer': fallback}

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node('elicit', self._node_elicit)
        graph.add_node('interpret', self._node_interpret)
        graph.add_node('baseline_retrieve', self._node_baseline_retrieve)
        graph.add_node('select_analysis', self._node_select_analysis)
        graph.add_node('plan_kg_from_context', self._node_plan_kg_from_context)
        graph.add_node('run_kg_query', self._node_run_kg_query)
        graph.add_node('run_analysis', self._node_run_analysis)
        graph.add_node('plan_pdf_from_kg', self._node_plan_pdf_from_kg)
        graph.add_node('pdf_refine', self._node_pdf_refine)
        graph.add_node('build_cycle_knowledge_package', self._node_build_cycle_knowledge_package)
        graph.add_node('integrate_state', self._node_integrate_state)
        graph.add_node('checkpoint', self._node_checkpoint)
        graph.add_node('vicarious_learning', self._node_vicarious_learning)
        graph.add_node('summarize_answer', self._node_summarize_answer)
        graph.add_node('build_knowledge_package', self._node_build_knowledge_package)
        graph.add_node('finalize', self._node_finalize)
        graph.add_node('pause_exit', self._node_pause_exit)

        def _route_entry(state: GraphState) -> str:
            resume = (state.get('resume_from') or '').strip()
            if resume in {
                'elicit',
                'interpret',
                'select_analysis',
                'vicarious_learning',
                'baseline_retrieve',
            }:
                self._emit(f'route START -> {resume} (resume)')
                return resume
            self._emit('route START -> elicit')
            return 'elicit'

        graph.add_conditional_edges(
            START,
            _route_entry,
            {
                'elicit': 'elicit',
                'interpret': 'interpret',
                'select_analysis': 'select_analysis',
                'vicarious_learning': 'vicarious_learning',
                'baseline_retrieve': 'baseline_retrieve',
            },
        )
        graph.add_conditional_edges(
            'elicit',
            self._route_after_elicit,
            {'interpret': 'interpret', 'pause_exit': 'pause_exit'},
        )
        graph.add_conditional_edges(
            'interpret',
            self._route_after_interpret,
            {'select_analysis': 'select_analysis', 'baseline_retrieve': 'baseline_retrieve'},
        )
        graph.add_edge('baseline_retrieve', 'summarize_answer')
        graph.add_edge('summarize_answer', 'build_knowledge_package')
        graph.add_edge('select_analysis', 'plan_kg_from_context')
        graph.add_edge('plan_kg_from_context', 'run_kg_query')
        graph.add_edge('run_kg_query', 'run_analysis')
        graph.add_edge('run_analysis', 'plan_pdf_from_kg')
        graph.add_edge('plan_pdf_from_kg', 'pdf_refine')
        graph.add_edge('pdf_refine', 'build_cycle_knowledge_package')
        graph.add_edge('build_cycle_knowledge_package', 'integrate_state')
        graph.add_edge('integrate_state', 'checkpoint')
        graph.add_conditional_edges(
            'checkpoint',
            self._route_after_checkpoint,
            {
                'select_analysis': 'select_analysis',
                'vicarious_learning': 'vicarious_learning',
                'pause_exit': 'pause_exit',
            },
        )
        graph.add_edge('vicarious_learning', 'build_knowledge_package')
        graph.add_edge('build_knowledge_package', 'finalize')
        graph.add_edge('finalize', END)
        graph.add_edge('pause_exit', END)
        return graph.compile()

    def run(
        self,
        question: str,
        *,
        question_embedding: Optional[List[float]] = None,
        use_kg: Optional[bool] = None,
        compare_baseline: bool = False,
        interactive: bool = False,
    ) -> OrchestratorResult:
        """Run the orchestrator pipeline.

        Args:
            question: The user question.
            question_embedding: Pre-computed embedding for the question.
            use_kg: Whether to use the knowledge graph (defaults to instance setting).
            compare_baseline: If True AND use_kg is enabled, also run a no-KG
                baseline and return both answers side-by-side. Defaults to False
                to avoid doubling latency and API cost.
            interactive: If True, pause for clarification / human collaboration gates.
        """
        self._emit('run start')
        effective_use_kg = self._use_kg if use_kg is None else bool(use_kg)
        primary = self._run_single(
            question=question,
            question_embedding=question_embedding,
            use_kg=effective_use_kg,
            interactive=interactive,
        )
        if primary.status != 'complete':
            self._emit(f'run paused status={primary.status} session={primary.session_id}')
            return primary
        if effective_use_kg and compare_baseline and not interactive:
            baseline = self._run_single(
                question=question,
                question_embedding=question_embedding,
                use_kg=False,
                interactive=False,
            )
            combined_answer = (
                f'Question: {question}\n\nVersion A (KG-augmented):\n{primary.answer.strip()}\n\n'
                f'Version B (No-KG baseline):\n{baseline.answer.strip()}'
            )
            self._emit(
                f'run done iterations_kg={primary.iterations_used} iterations_baseline={baseline.iterations_used}'
            )
            return OrchestratorResult(
                answer=combined_answer,
                state=primary.state,
                iterations_used=primary.iterations_used + baseline.iterations_used,
                status='complete',
            )
        self._emit(f'run done iterations={primary.iterations_used}')
        return primary

    def continue_session(
        self,
        *,
        session_id: str,
        clarification_answers: Optional[str] = None,
        human_feedback: Optional[str] = None,
        question_embedding: Optional[List[float]] = None,
    ) -> OrchestratorResult:
        session = self._session_store.pop(session_id)
        if session is None:
            raise ValueError(f'Unknown or expired session_id: {session_id}')
        state = dict(session.graph_state)
        state['pause_status'] = None
        state['await_human_gate'] = False
        state['collaboration_prompt'] = None
        state['interactive'] = session.interactive
        if question_embedding is not None:
            state['question_embedding'] = question_embedding

        if session.phase == 'elicitation':
            answers = (clarification_answers or human_feedback or '').strip()
            if not answers:
                raise ValueError('clarification_answers required to continue elicitation.')
            base_q = state.get('question') or ''
            merged_prompt = (
                f'{base_q}\n\nClarifications from researcher:\n{answers}\n\n'
                f'Prior clarifying questions: {session.clarifying_questions}'
            )
            intent = self.elicitation.elicit(merged_prompt)
            # After clarification, proceed even if still imperfectly specified.
            intent = intent.model_copy(update={'is_sufficiently_specified': True})
            state['research_intent'] = intent.model_dump()
            state['question'] = intent.refined_question or base_q
            if state.get('question_embedding') is None and state['question']:
                state['question_embedding'] = self._embed_fn([state['question']])[0]
            state['resume_from'] = 'interpret'
            scratch = list(state.get('scratchpad') or [])
            scratch.append(f'[elicitation_resume] answers={answers[:300]!r}')
            state['scratchpad'] = scratch
        elif session.phase == 'human_collaboration':
            feedback = (human_feedback or clarification_answers or 'continue').strip()
            fb_lower = feedback.lower()
            step = int(state.get('analysis_step', 0))
            if fb_lower in {'stop', 'done', 'finish', 'sufficient'}:
                state['stop_gathering'] = True
                state['resume_from'] = 'vicarious_learning'
                scratch = list(state.get('scratchpad') or [])
                scratch.append('[human_collaboration] decision=stop')
                state['scratchpad'] = scratch
            else:
                state['stop_gathering'] = False
                state['analysis_step'] = step + 1
                state['human_feedback'] = feedback
                memo = state.get('integration_memo') or ''
                if fb_lower not in {'continue', 'yes', 'y', 'ok', 'proceed'}:
                    memo = f'{memo}\n[human_feedback] {feedback}'.strip()
                    state['integration_note'] = feedback[:300]
                state['integration_memo'] = memo
                state['resume_from'] = 'select_analysis'
                scratch = list(state.get('scratchpad') or [])
                scratch.append(f'[human_collaboration] decision=continue feedback={feedback[:200]!r}')
                state['scratchpad'] = scratch
        else:
            raise ValueError(f'Unsupported session phase: {session.phase}')

        final_graph_state = self._graph.invoke(state)
        return self._result_from_graph_state(
            final_graph_state,
            interactive=session.interactive,
            use_kg=session.use_kg,
            compare_baseline=session.compare_baseline,
            existing_session_id=None,
        )
