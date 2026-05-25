from __future__ import annotations
import logging
from typing import Any, Dict, List
from .models import SummaryOutput

logger = logging.getLogger(__name__)


class SummarizingAgent:

    def __init__(self, client: Any, model: str='gpt-5.2') -> None:
        self._client = client
        self._model = model

    def summarize(self, records: List[Dict[str, Any]], *, question: str='', max_records_in_prompt: int=40, max_chars_per_record: int=4000) -> SummaryOutput:
        unique_dois: List[str] = []
        formatted_chunks: List[str] = []
        for i, r in enumerate(records[:max_records_in_prompt], start=1):
            chunk_lines = [f'--- SOURCE {i} ---']
            doi = r.get('doi')
            if isinstance(doi, str) and doi.strip() and (doi.strip() not in unique_dois):
                unique_dois.append(doi.strip())
            for k, v in r.items():
                if not isinstance(v, str) or not v.strip():
                    continue
                if len(v) > max_chars_per_record:
                    v = v[:max_chars_per_record].rsplit(' ', 1)[0] + '...'
                chunk_lines.append(f'{k.upper()}:\n{v}')
            formatted_chunks.append('\n'.join(chunk_lines))
        sources_block = '\n\n'.join(formatted_chunks) if formatted_chunks else '(no chunk text)'
        q = (question or '').strip()
        user_content = f'## User question (answer with this intent)\n{q or '(not provided)'}\n\n## Extracted publication chunks\n{sources_block}\n\n## Instructions\nProduce a concise summary of the chunks **focused on what is needed to answer the user question**. Prioritize passages that bear on the question; de-emphasize unrelated material. Use plain text or light Markdown. Cite DOIs inline where possible.'
        system = "You summarize extracted PDF chunks with a clear intent: support answering the user's question. Do not write a generic literature review unless the question is broad. If chunks are off-topic for the question, say so briefly and summarize only what is relevant."
        try:
            response = self._client.generate(model=self._model, system=system, prompt=user_content, stream=False)
            summary_text = response.get('response', '').strip()
            return SummaryOutput(summary=summary_text, cited_dois=unique_dois)
        except Exception as exc:
            logger.warning('Summarization failed: %s: %s', type(exc).__name__, exc)
            lines: List[str] = []
            for row in records[:5]:
                text = row.get('text', '')
                if isinstance(text, str) and text.strip():
                    safe_text = text.strip()[:220].rsplit(' ', 1)[0]
                    lines.append(f'{safe_text}...')
            fallback_summary = ' '.join(lines).strip() or 'Local model generation failed.'
            return SummaryOutput(summary=f'**Question:** {q or "(not provided)"}\n\n**Partial evidence:** {fallback_summary}', cited_dois=unique_dois)
