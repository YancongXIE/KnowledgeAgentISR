from __future__ import annotations
import json
import logging
from typing import Any, Dict, List
from pydantic import ValidationError
from .llm_json import complete_json
from .models import IntegrationOutput

logger = logging.getLogger(__name__)

def _safe_json(value: Any, *, max_chars: int=12000) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)[:max_chars]
    except Exception:
        return str(value)[:max_chars]

def _window_text(text: str, *, max_chars: int) -> str:
    value = (text or '').strip()
    if len(value) <= max_chars:
        return value
    marker = '\n\n... *[truncated]* ...\n\n'
    head = max(20, int(max_chars * 0.45))
    tail = max(20, max_chars - head - len(marker))
    return f'{value[:head]}{marker}{value[-tail:]}'

def _kg_rows_markdown(rows: List[Dict[str, Any]], *, max_rows: int) -> str:
    if not rows:
        return '*No graph rows.*'
    lines: List[str] = []
    for i, row in enumerate(rows[:max_rows], start=1):
        try:
            snippet = json.dumps(row, ensure_ascii=False)[:800]
        except (TypeError, ValueError):
            snippet = str(row)[:800]
        lines.append(f'{i}. `{snippet}`')
    return '\n'.join(lines)

def _build_integrate_user_markdown(*, stage: str, question: str, pdf_summary: str, integration_memo: str, kg_records: List[Dict[str, Any]], analysis_report: Dict[str, Any] | None, analysis_note: str, max_kg_rows: int) -> str:
    excerpt = _safe_json(analysis_report, max_chars=10000)
    return '\n'.join(['## User question', (question or '(not provided)').strip(), '', '## Pipeline stage', stage, '', '## PDF / chunk evidence summary', _window_text(pdf_summary, max_chars=5000), '', '## Integration memo (running context)', _window_text(integration_memo, max_chars=5000), '', '## Knowledge graph sample rows', _kg_rows_markdown(kg_records, max_rows=max_kg_rows), '', '## Analysis note', _window_text(analysis_note, max_chars=1000), '', '## Analysis report excerpt', 'Use only as factual support; do not invent beyond it.', '', '```json', excerpt, '```', '', '## Your task', 'Return a single JSON object matching the required schema. `merged_context` should integrate the above faithfully and stay centered on the user question.'])

def _build_compose_final_user_markdown(*, question: str, integration_memo: str, pdf_summary: str, kg_records: List[Dict[str, Any]], analysis_note: str, analysis_report: Dict[str, Any] | None, max_kg_rows: int) -> str:
    excerpt = _safe_json(analysis_report, max_chars=9000)
    return '\n'.join(['## User question (answer this directly)', (question or '(not provided)').strip(), '', '## Integrated context', _window_text(integration_memo, max_chars=5000), '', '## PDF evidence summary', _window_text(pdf_summary, max_chars=4500), '', '## Knowledge graph rows', _kg_rows_markdown(kg_records, max_rows=max_kg_rows), '', '## Analysis note', _window_text(analysis_note, max_chars=700), '', '## Analysis report excerpt', '```json', excerpt, '```', '', '## Output requirements', '- Write **only** the answer text (no JSON).', '- **First sentence** must address the user question directly.', '- At most two short paragraphs.', '- No follow-up questions or suggested tasks.', '- If evidence is insufficient, say so in one sentence, then give the best partial answer.'])

class IntegrationAgent:

    def __init__(self, client: Any, model: str='gpt-5.2') -> None:
        self._client = client
        self._model = model

    def integrate(self, *, stage: str, question: str, pdf_summary: str, integration_memo: str, kg_records: List[Dict[str, Any]], analysis_report: Dict[str, Any] | None, analysis_note: str, max_kg_rows: int=12) -> IntegrationOutput:
        system = 'You integrate state across a KG+LLM pipeline. Produce merged context that is faithful to evidence and centered on the user question. Do not invent facts. Keep it concise and useful for the next step.'
        user = _build_integrate_user_markdown(stage=stage, question=question, pdf_summary=pdf_summary, integration_memo=integration_memo, kg_records=kg_records, analysis_report=analysis_report, analysis_note=analysis_note, max_kg_rows=max_kg_rows)
        try:
            return complete_json(self._client, model=self._model, system=system, user=user, schema_model=IntegrationOutput)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.debug('Integration LLM returned invalid output: %s', exc)
        except Exception as exc:
            logger.warning('Integration LLM failed: %s: %s', type(exc).__name__, exc)
        merged = (integration_memo or '').strip()
        if pdf_summary.strip():
            merged = f'{merged}\n{pdf_summary.strip()}'.strip() if merged else pdf_summary.strip()
        return IntegrationOutput(stage=stage, merged_context=f'**Question:** {(question or "").strip() or "(not provided)"}\n\n**Best-effort context:**\n{merged}', key_points=['integration_fallback'], next_focus='')

    @staticmethod
    def _central_constructs_from_analysis(analysis_report: Dict[str, Any] | None, *, max_items: int=5) -> List[str]:
        if not isinstance(analysis_report, dict):
            return []
        l2 = analysis_report.get('level_2_relationship_mining')
        if not isinstance(l2, dict):
            return []
        names: List[str] = []
        for key, _metric in (('indegreecentrality', 'Indegree'), ('outdegreecentrality', 'Outdegree'), ('betweennesscentrality', 'Betweenness')):
            node = l2.get(key)
            if not isinstance(node, dict):
                continue
            ranking = node.get('ranking')
            if not isinstance(ranking, list):
                continue
            for row in ranking[:max_items]:
                if not isinstance(row, dict):
                    continue
                concept = row.get('Concept')
                if isinstance(concept, str) and concept.strip():
                    names.append(concept.strip())
        deduped = list(dict.fromkeys(names))
        return deduped[:max_items]

    def compose_final_answer(self, *, question: str, pdf_summary: str, integration_memo: str, kg_records: List[Dict[str, Any]], analysis_report: Dict[str, Any] | None, analysis_note: str, max_kg_rows: int=10) -> str:
        system = "You answer exactly the user's question using the evidence sections below. Do not write a generic summary of sources — synthesize into a direct answer. Stay on-topic. Do not ask follow-up questions. If evidence is weak, say so briefly."
        prompt = _build_compose_final_user_markdown(question=question, integration_memo=integration_memo, pdf_summary=pdf_summary, kg_records=kg_records, analysis_note=analysis_note, analysis_report=analysis_report, max_kg_rows=max_kg_rows)
        try:
            response = self._client.generate(model=self._model, system=system, prompt=prompt, stream=False)
            text = (response.get('response', '') if isinstance(response, dict) else '').strip()
            if text:
                return text
        except Exception as exc:
            q = (question or '').strip()
            central = self._central_constructs_from_analysis(analysis_report, max_items=5)
            if central:
                return f'**Answer:** The constructs that appear most central in the graph (by centrality rankings) include: {', '.join(central)}.\n\n*(Final-answer LLM call failed: {type(exc).__name__}: {exc})*'
            if pdf_summary.strip():
                return f'**Answer (from PDF evidence; model call failed):** {pdf_summary.strip()[:2000]}\n\n*(Error: {type(exc).__name__}: {exc})*'
            return f'**Question:** {q or '(not provided)'}\n\n**Answer:** Insufficient evidence after an API error ({type(exc).__name__}: {exc}).'
        central = self._central_constructs_from_analysis(analysis_report, max_items=5)
        if central:
            return f'Most central constructs in the trust models are: {', '.join(central)}. This is based on graph centrality signals (indegree/outdegree/betweenness) from the KG.'
        if pdf_summary.strip():
            return pdf_summary.strip()
        return 'Insufficient evidence to answer confidently.'
