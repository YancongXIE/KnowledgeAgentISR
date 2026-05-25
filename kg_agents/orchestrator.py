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
from .extraction_agent import ExtractionAgent
from .graph_query_agent import GraphQueryAgent
from .integration_agent import IntegrationAgent
from .llm_json import complete_json
from .models import RetrievalPlan
from .schema_agent import SchemaAgent
from .state import AgentState
from .summarizing_agent import SummarizingAgent

logger = logging.getLogger(__name__)

class _PdfChunkFocusJson(BaseModel):
    retrieval_instruction: str = Field(description='Natural-language instruction for the next PDF/chunk vector retrieval pass')
    rationale: str = Field(default='', description='Why this PDF focus helps answer the question')

class _CheckpointSufficiencyJson(BaseModel):
    sufficient: bool = Field(description="True only if the evidence below is enough to answer the user's question adequately.")
    rationale: str = Field(default='', description='One or two sentences; reference the question.')

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

class KGMultiAgentOrchestrator:
    ACTION_CATALOG: Dict[str, str] = {'plan_kg_from_context': 'Decide KG retrieval focus from current context', 'run_kg_query': 'Execute read-only Cypher against KG', 'run_analysis': 'Run analysis capabilities on KG query results', 'plan_pdf_from_kg': 'Decide next PDF chunk focus from KG results', 'pdf_refine': 'Vector-retrieve chunks using KG-informed instruction', 'summarize_answer': 'Summarize evidence and produce final answer'}
    ANALYSIS_CAPABILITIES: Dict[str, List[Dict[str, str]]] = {'concept': [{'function_name': 'related_publications', 'description': 'This function identifies the publications that are related to the concept.'}, {'function_name': 'definitions', 'description': 'This function identifies the definitions of the concept.'}, {'function_name': 'definition_similarity', 'description': 'This function calculates the similarity score among concept definitions.'}, {'function_name': 'related_theories', 'description': 'This function identifies the theories that are related to the concept.'}], 'relationship': [{'function_name': 'antecedents_consequents', 'description': 'This function identifies the antecedents and consequents of a concept.'}, {'function_name': 'mediators_moderators', 'description': 'This function identifies the mediators and moderators of a relationship.'}, {'function_name': 'indegreecentrality', 'description': 'This function calculates the indegree centrality of a concept. Higher indegree centrality indicates more incoming connections, and popular consequents.'}, {'function_name': 'outdegreecentrality', 'description': 'This function calculates the outdegree centrality of a concept. Higher outdegree centrality indicates more outgoing connections, and fundamental antecedents.'}, {'function_name': 'betweennesscentrality', 'description': 'This function calculates the betweenness centrality of a concept. Higher betweenness centrality indicates influence as a bridge between other concepts.'}, {'function_name': 'cutpoints', 'description': 'This function identifies the cutpoints of a relationship. Cutpoints are concepts that, if removed, would disconnect the graph.'}, {'function_name': 'periphery', 'description': 'This function identifies the periphery index of a concept. Higher periphery index indicates concepts are close to the edge of the graph and more peripheral or innovative.'}, {'function_name': 'structural_hole_measures', 'description': 'This function calculates the structural hole measures of a concept. Constraint and effective size are calculated.'}, {'function_name': 'association_rules', 'description': 'This function identifies concepts tend to occur together in the same conceptual model.'}, {'function_name': 'knowledge_index', 'description': 'Knowledge index (KI) reflects conceptual convergence. Higher KI values mean antecedent paths are more convergent. A few antecedent paths dominate the explanation of focal dependent concept.Lower KI values mean antecedent paths are more divergent.'}]}
    _MEMO_ANCHOR_PREFIXES: tuple[str, ...] = ('[analysis_queue_seed]', '[select_analysis]', '[cycle_objective]', '[checkpoint]')

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

    def __init__(self, driver: Driver, llm_client: Any, embed_fn: Callable[[List[str]], List[List[float]]], *, model: str='gpt-5.2', max_iterations: int=4, max_cycles: int=4, default_candidate_pool: int=100, log_progress: bool=False, log_sink: Optional[Callable[[str], None]]=None, use_kg: bool=True) -> None:
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
        self.schema = SchemaAgent()
        self.graph_query = GraphQueryAgent(llm_client, self.schema, model=model)
        self.extraction = ExtractionAgent(embed_fn)
        _curation = os.environ.get('ANALYSIS_CURATION_ID', '13')
        self.analyzing = AnalyzingAgent(embed_fn, driver=driver, curation_id=str(_curation))
        self.summarizing = SummarizingAgent(llm_client, model=model)
        self.integration = IntegrationAgent(llm_client, model=model)
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

    def _initial_graph_state(self, *, question: str, question_embedding: Optional[List[float]], use_kg: bool) -> GraphState:
        return {'question': question, 'question_embedding': question_embedding, 'scratchpad': [], 'available_actions': list(self.ACTION_CATALOG.keys()), 'selected_actions': [], 'completed_actions': [], 'interpretation': None, 'retrieval_plan': None, 'last_cypher': None, 'last_records': [], 'extracted_records': [], 'last_error': None, 'analysis_report': None, 'analysis_note': None, 'evidence_summary': None, 'summary_cited_dois': [], 'integration_memo': '', 'integration_trace': [], 'iteration': 0, 'extra': None, 'continue_loop': True, 'final_answer': None, 'integration_note': '', 'pdf_focus_instruction': None, 'cycle_index': 0, 'selected_analyses': [], 'analysis_queue': [], 'analysis_step': 0, 'stop_gathering': False, 'use_kg': use_kg}

    def _run_single(self, *, question: str, question_embedding: Optional[List[float]], use_kg: bool) -> OrchestratorResult:
        initial_state = self._initial_graph_state(question=question, question_embedding=question_embedding, use_kg=use_kg)
        final_graph_state = self._graph.invoke(initial_state)
        final_state = self._agent_state_from_graph(final_graph_state)
        final_answer = final_graph_state.get('final_answer') or self._compose_final_answer(final_state, final_state.evidence_summary or '')
        return OrchestratorResult(answer=final_answer, state=final_state, iterations_used=final_state.iteration)

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
        system = 'Select the SINGLE best next Analysaurus analysis function (by function_name). The catalog in analysis_capabilities lists each function_name and description. Choose functions by interpreting the MEANING and PURPOSE of each analysis description, not by lexical overlap with the question text. Use current state, prior analyses, and evidence summaries to decide what is most useful now. Avoid repeating analyses that were already run unless there is a strong reason from the evidence. Return JSON only with selected_analyses (first item is the next analysis), objective, rationale.'
        user = json.dumps({'question': state['question'], 'cycle_index': cycle_index, 'already_run_analyses': already_run, 'analysis_capabilities': str(self.ANALYSIS_CAPABILITIES), 'analysis_brief': self._analysis_brief(state.get('analysis_report')), 'kg_rows_excerpt': _kg_snippets(list(state.get('last_records', []))), 'pdf_chunks_excerpt': _chunk_snippets(list(state.get('extracted_records', []))), 'context_memo': self._memo_window(state.get('integration_memo') or '', max_chars=2500)}, ensure_ascii=False, indent=2)
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

    def _plan_kg(self, *, question: str, found_records: List[dict], integration_memo: str, selected_analyses: List[str], selected_levels: List[str], objective: str) -> RetrievalPlan:
        system = 'You plan what to retrieve from a Neo4j knowledge graph (concepts, relations, publications) based on evidence found so far. Output JSON matching the required keys. Do not write Cypher.\n\n' + self.schema.prompt_fragment()
        user = json.dumps({'user_question': question, 'evidence_excerpts': _chunk_snippets(found_records), 'selected_analyses': selected_analyses, 'selected_levels': selected_levels, 'cycle_objective': objective, 'running_context_memo': self._memo_window(integration_memo, max_chars=2200)}, ensure_ascii=False, indent=2)
        try:
            return complete_json(self._llm_client, model=self._model, system=system, user=user, schema_model=RetrievalPlan)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('KG plan LLM returned invalid schema: %s', exc)
        except Exception as exc:
            logger.warning('KG plan LLM failed: %s: %s', type(exc).__name__, exc)
        q = question.lower()
        rels = ['IS_ANTECEDENT_OF', 'IS_CONSEQUENT_OF'] if 'trust' in q else []
        return RetrievalPlan(intent='Iterative fallback: explore structure and publication-linked chunks.', target_labels=['Publication', 'Chunk', 'Element'], key_properties={}, preferred_relationships=rels, search_hint='Use graph patterns and vector search over chunks as needed.')

    def _plan_pdf(self, *, question: str, found_records: List[dict], integration_memo: str) -> str:
        system = 'You decide what to search next in PDF text chunks (vector retrieval) based on evidence found so far (concepts, relations, definitions, DOIs, and other retrieved context). Return a focused natural-language retrieval_instruction for finding the most useful passages.'
        user = json.dumps({'user_question': question, 'found_evidence_sample': _kg_snippets(found_records), 'running_context_memo': self._memo_window(integration_memo, max_chars=2200)}, ensure_ascii=False, indent=2)
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

    def _node_interpret(self, state: GraphState) -> GraphState:
        self._emit('node=interpret start')
        interpretation = {'strategy': 'adaptive', 'note': 'No fixed query-type heuristic; planning is driven by iterative evidence.'}
        self._emit('node=interpret done strategy=adaptive')
        scratch = self._append_log(state, '[interpret] strategy=adaptive')
        scratch.append('[analysis_queue] empty at start; selector chooses next analysis each cycle')
        return {'available_actions': list(self.ACTION_CATALOG.keys()), 'interpretation': interpretation, 'analysis_queue': [], 'analysis_step': 0, 'stop_gathering': False, 'selected_analyses': [], 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state), 'use_kg': bool(state.get('use_kg', self._use_kg))}

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
        plan = self._plan_kg(question=state['question'], found_records=list(state.get('extracted_records', [])), integration_memo=state.get('integration_memo', ''), selected_analyses=analyses, selected_levels=levels, objective=objective)
        plan = self._enrich_plan_with_objective(plan, objective, cycle, levels)
        scratch = self._append_log(state, f'[plan_kg_from_context] cycle={cycle} objective={objective}')
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
        question = state['question']
        question_embedding = state.get('question_embedding')
        rp = state.get('retrieval_plan') or {}
        plan = RetrievalPlan.model_validate(rp)
        bundle = self.graph_query.propose_query(question, plan, question_embedding=question_embedding, extra_context=self._memo_window(state.get('integration_memo', ''), max_chars=1800))
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
        objective = self._cycle_objective(state)
        instruction = self._plan_pdf(question=state['question'], found_records=list(state.get('last_records', [])), integration_memo=state.get('integration_memo', ''))
        scratch = self._append_log(state, f'[plan_pdf_from_kg] instruction={instruction[:120]!r}...')
        self._emit('node=plan_pdf_from_kg done')
        return {'pdf_focus_instruction': instruction, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_pdf_refine(self, state: GraphState) -> GraphState:
        self._emit('node=pdf_refine start')
        question = state['question']
        step = int(state.get('analysis_step', 0))
        instr = (state.get('pdf_focus_instruction') or question).strip()
        new_rows, note = self.extraction.extract_from_instruction(self._driver, instr, question=question, question_embedding=state.get('question_embedding'), fallback_records=list(state.get('last_records', [])))
        merged = self._merge_chunk_rows(list(state.get('extracted_records', [])), new_rows)
        scratch = self._append_log(state, f'[pdf_refine] analysis_step={step} {note}')
        self._emit(f'node=pdf_refine done analysis_step={step} total_chunks={len(merged)}')
        return {'extracted_records': merged, 'cycle_index': step, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_integrate_state(self, state: GraphState) -> GraphState:
        self._emit('node=integrate_state start')
        out = self.integration.integrate(stage=f'cycle_{int(state.get('analysis_step', 0))}', question=state.get('question', ''), pdf_summary=_chunk_snippets(list(state.get('extracted_records', [])), max_chunks=8, max_chars=600), integration_memo=self._memo_window(state.get('integration_memo', ''), max_chars=5000), kg_records=list(state.get('last_records', [])), analysis_report=state.get('analysis_report'), analysis_note=state.get('analysis_note') or '')
        merged_context = (out.merged_context or '').strip()
        memo = merged_context or state.get('integration_memo', '')
        trace = list(state.get('integration_trace', []))
        trace.append({'stage': out.stage or 'integrate_state', 'analysis_step': int(state.get('analysis_step', 0)), 'key_points': list(out.key_points or [])[:8], 'next_focus': (out.next_focus or '')[:300]})
        scratch = self._append_log(state, f'[integrate_state] key_points={len(out.key_points or [])} next_focus={(out.next_focus or '')[:120]!r}')
        self._emit('node=integrate_state done')
        return {'integration_memo': memo, 'integration_trace': trace, 'integration_note': out.next_focus or '', 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _checkpoint_evidence_block(self, state: GraphState) -> str:
        q = (state.get('question') or '').strip()
        memo = self._memo_window(state.get('integration_memo') or '', max_chars=3200)
        ab = self._analysis_brief(state.get('analysis_report'))
        chunks = _chunk_snippets(list(state.get('extracted_records', [])), max_chunks=6, max_chars=500)
        kg = _kg_snippets(list(state.get('last_records', [])), max_rows=4, max_chars=400)
        step = int(state.get('analysis_step', 0))
        queue = list(state.get('analysis_queue') or [])
        cur = queue[step] if queue and 0 <= step < len(queue) else '(unknown)'
        return f'USER QUESTION (this is the only goal to satisfy):\n{q}\n\nCurrent analysis function for this iteration: {cur}\n\nRunning integration memo:\n{memo}\n\nAnalysis brief (this cycle):\n{ab or '(none)'}\n\nLatest graph sample:\n{kg}\n\nPDF chunks gathered so far (excerpt):\n{chunks}\n'

    def _evaluate_sufficiency(self, state: GraphState) -> tuple[bool, str]:
        """Ask the LLM whether evidence is sufficient. Returns (sufficient, rationale)."""
        system = 'You judge whether the accumulated evidence answers the USER QUESTION. The user question is the sole success criterion — stay on topic. Answer sufficient=true only if a reasonable answer can be given from this evidence (definitions, relations, or PDF excerpts) without important gaps. If the evidence is empty, off-topic, or too weak, answer sufficient=false. Return JSON only with keys sufficient and rationale.'
        user = self._checkpoint_evidence_block(state)
        try:
            out = complete_json(self._llm_client, model=self._model, system=system, user=user, schema_model=_CheckpointSufficiencyJson)
            return bool(out.sufficient), (out.rationale or '').strip()
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('Checkpoint LLM returned invalid schema: %s', exc)
            return False, 'Checkpoint LLM returned invalid JSON; continuing.'
        except Exception as exc:
            logger.warning('Checkpoint LLM failed: %s: %s', type(exc).__name__, exc)
            return False, 'Checkpoint LLM failed; continuing if more analyses remain.'

    def _node_checkpoint(self, state: GraphState) -> GraphState:
        self._emit('node=checkpoint start')
        step = int(state.get('analysis_step', 0))
        next_step = step + 1
        exhausted = next_step >= self._max_cycles

        sufficient, rationale = self._evaluate_sufficiency(state)
        should_stop = sufficient or exhausted

        if sufficient:
            self._emit('node=checkpoint done sufficient=True -> summarize')
        elif exhausted:
            self._emit(f'node=checkpoint done exhausted step={step} -> summarize')
        else:
            self._emit(f'node=checkpoint done continue -> analysis_step={next_step}')
        trace = list(state.get('integration_trace', []))
        trace.append({'stage': 'checkpoint', 'analysis_step': step, 'sufficient': sufficient, 'exhausted': exhausted, 'rationale': rationale[:500]})
        scratch = self._append_log(state, f'[checkpoint] sufficient={sufficient} exhausted={exhausted} next_step={next_step} stop={should_stop}')
        out_state: GraphState = {'stop_gathering': should_stop, 'integration_memo': state.get('integration_memo', ''), 'scratchpad': scratch, 'integration_trace': trace, 'iteration': self._bump_iteration(state)}
        if not should_stop:
            out_state['analysis_step'] = next_step
        return out_state

    def _route_after_checkpoint(self, state: GraphState) -> str:
        if state.get('stop_gathering'):
            self._emit('route checkpoint -> summarize_answer')
            return 'summarize_answer'
        self._emit('route checkpoint -> select_analysis')
        return 'select_analysis'

    def _prepare_analysis_for_summary(self, state: GraphState) -> tuple[Dict[str, Any] | None, str]:
        """Ensure we have an analysis report for the summarize step."""
        report = state.get('analysis_report')
        anote = state.get('analysis_note') or ''
        if report is None and state.get('last_records'):
            report, anote = self.analyzing.analyze(list(state.get('last_records', [])))
        return report, anote

    def _build_final_answer(self, state: GraphState, summary_out: Any, integrated: Any, report: Any, anote: str) -> str:
        """Compose the final answer text from integration and summary outputs."""
        integrated_text = (integrated.merged_context or '').strip()
        final_text = self.integration.compose_final_answer(
            question=state.get('question', ''),
            pdf_summary=summary_out.summary,
            integration_memo=integrated_text or state.get('integration_memo', ''),
            kg_records=list(state.get('last_records', [])),
            analysis_report=report,
            analysis_note=anote,
        )
        if not final_text.strip():
            final_text = integrated_text or summary_out.summary
        merged: GraphState = dict(state)
        merged['evidence_summary'] = summary_out.summary
        merged['summary_cited_dois'] = list(summary_out.cited_dois)
        merged['analysis_report'] = report
        merged['analysis_note'] = anote
        merged['integration_memo'] = integrated_text or state.get('integration_memo', '')
        agent_state = self._agent_state_from_graph(merged)
        return self._ensure_question_anchor(
            state.get('question', ''),
            self._compose_final_answer(agent_state, final_text),
        )

    def _node_summarize_answer(self, state: GraphState) -> GraphState:
        self._emit('node=summarize_answer start')
        report, anote = self._prepare_analysis_for_summary(state)
        summary_out = self.summarizing.summarize(list(state.get('extracted_records', [])), question=state.get('question', ''))
        integrated = self.integration.integrate(
            stage='final_answer',
            question=state.get('question', ''),
            pdf_summary=summary_out.summary,
            integration_memo=state.get('integration_memo', ''),
            kg_records=list(state.get('last_records', [])),
            analysis_report=report,
            analysis_note=anote,
        )
        integrated_text = (integrated.merged_context or '').strip()
        final = self._build_final_answer(state, summary_out, integrated, report, anote)
        memo = integrated_text or state.get('integration_memo', '')
        scratch = self._append_log(state, f'[summarize] len={len(summary_out.summary)} dois={len(summary_out.cited_dois)}')
        trace = list(state.get('integration_trace', []))
        trace.append({'stage': integrated.stage or 'final_answer', 'analysis_step': int(state.get('analysis_step', 0)), 'key_points': list(integrated.key_points or [])[:8], 'next_focus': (integrated.next_focus or '')[:300]})
        self._emit('node=summarize_answer done')
        return {'analysis_report': report, 'analysis_note': anote, 'evidence_summary': summary_out.summary, 'summary_cited_dois': list(summary_out.cited_dois), 'final_answer': final, 'integration_memo': memo, 'integration_trace': trace, 'integration_note': integrated.next_focus or '', 'scratchpad': scratch, 'iteration': self._bump_iteration(state)}

    def _node_finalize(self, state: GraphState) -> GraphState:
        self._emit('node=finalize')
        if state.get('final_answer'):
            return {}
        fallback = state.get('evidence_summary') or 'No sufficient evidence gathered.'
        return {'final_answer': fallback}

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node('interpret', self._node_interpret)
        graph.add_node('baseline_retrieve', self._node_baseline_retrieve)
        graph.add_node('select_analysis', self._node_select_analysis)
        graph.add_node('plan_kg_from_context', self._node_plan_kg_from_context)
        graph.add_node('run_kg_query', self._node_run_kg_query)
        graph.add_node('run_analysis', self._node_run_analysis)
        graph.add_node('plan_pdf_from_kg', self._node_plan_pdf_from_kg)
        graph.add_node('pdf_refine', self._node_pdf_refine)
        graph.add_node('integrate_state', self._node_integrate_state)
        graph.add_node('checkpoint', self._node_checkpoint)
        graph.add_node('summarize_answer', self._node_summarize_answer)
        graph.add_node('finalize', self._node_finalize)
        graph.add_edge(START, 'interpret')
        graph.add_conditional_edges('interpret', self._route_after_interpret, {'select_analysis': 'select_analysis', 'baseline_retrieve': 'baseline_retrieve'})
        graph.add_edge('baseline_retrieve', 'summarize_answer')
        graph.add_edge('select_analysis', 'plan_kg_from_context')
        graph.add_edge('plan_kg_from_context', 'run_kg_query')
        graph.add_edge('run_kg_query', 'run_analysis')
        graph.add_edge('run_analysis', 'plan_pdf_from_kg')
        graph.add_edge('plan_pdf_from_kg', 'pdf_refine')
        graph.add_edge('pdf_refine', 'integrate_state')
        graph.add_edge('integrate_state', 'checkpoint')
        graph.add_conditional_edges('checkpoint', self._route_after_checkpoint, {'select_analysis': 'select_analysis', 'summarize_answer': 'summarize_answer'})
        graph.add_edge('summarize_answer', 'finalize')
        graph.add_edge('finalize', END)
        return graph.compile()

    def run(self, question: str, *, question_embedding: Optional[List[float]]=None, use_kg: Optional[bool]=None, compare_baseline: bool=False) -> OrchestratorResult:
        """Run the orchestrator pipeline.

        Args:
            question: The user question.
            question_embedding: Pre-computed embedding for the question.
            use_kg: Whether to use the knowledge graph (defaults to instance setting).
            compare_baseline: If True AND use_kg is enabled, also run a no-KG
                baseline and return both answers side-by-side. Defaults to False
                to avoid doubling latency and API cost.
        """
        self._emit('run start')
        effective_use_kg = self._use_kg if use_kg is None else bool(use_kg)
        primary = self._run_single(question=question, question_embedding=question_embedding, use_kg=effective_use_kg)
        if effective_use_kg and compare_baseline:
            baseline = self._run_single(question=question, question_embedding=question_embedding, use_kg=False)
            combined_answer = f'Question: {question}\n\nVersion A (KG-augmented):\n{primary.answer.strip()}\n\nVersion B (No-KG baseline):\n{baseline.answer.strip()}'
            self._emit(f'run done iterations_kg={primary.iterations_used} iterations_baseline={baseline.iterations_used}')
            return OrchestratorResult(answer=combined_answer, state=primary.state, iterations_used=primary.iterations_used + baseline.iterations_used)
        self._emit(f'run done iterations={primary.iterations_used}')
        return primary
