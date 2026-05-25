from __future__ import annotations
from typing import Any, Callable, Dict, List, Sequence
import numpy as np
from neo4j import Driver
_CHUNK_TEXT_KEYS: Sequence[str] = ('text', 'chunk_text', 'passage')

def _first_chunk_text(row: Dict[str, Any]) -> str | None:
    for key in _CHUNK_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

class ExtractionAgent:

    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]], index_name: str='chunkEmb', candidate_pool: int=100, top_k_chunks: int=10) -> None:
        self._embed = embed_fn
        self._index_name = index_name
        self._candidate_pool = candidate_pool
        self._top_k_chunks = top_k_chunks

    def extract_from_instruction(self, driver: Driver, instruction: str, *, question: str, question_embedding: List[float] | None=None, fallback_records: List[Dict[str, Any]] | None=None) -> tuple[List[Dict[str, Any]], str]:
        text_instruction = (instruction or '').strip() or question
        try:
            if question_embedding is not None and text_instruction == question:
                emb = np.asarray(question_embedding, dtype=np.float64).tolist()
            else:
                emb = np.asarray(self._embed([text_instruction])[0], dtype=np.float64).tolist()
            cypher = '\n            CALL db.index.vector.queryNodes($index_name, $candidate_pool, $embedding)\n            YIELD node, score\n            OPTIONAL MATCH (node)<-[:HAS_CHUNK]-(p:Publication)\n            RETURN node.text AS text,\n                   score AS score,\n                   node.uuid AS chunk_uuid,\n                   node.number AS chunk_number,\n                   p.DOI AS doi\n            ORDER BY score DESC\n            LIMIT $k\n            '
            records, _, _ = driver.execute_query(cypher, index_name=self._index_name, candidate_pool=self._candidate_pool, embedding=emb, k=self._top_k_chunks)
            extracted = [dict(r) for r in records]
            if extracted:
                return (extracted, f"extraction: vector query selected {len(extracted)} chunks (instruction='{text_instruction[:80]}')")
            if fallback_records is not None:
                fallback_chunks, fallback_note = self.extract_from_rows(question, fallback_records, question_embedding=question_embedding)
                return (fallback_chunks, 'extraction: vector query returned 0 rows; ' + fallback_note)
            return ([], 'extraction: vector query returned 0 rows')
        except Exception as exc:
            if fallback_records is not None:
                fallback_chunks, fallback_note = self.extract_from_rows(question, fallback_records, question_embedding=question_embedding)
                return (fallback_chunks, f'extraction: vector query failed ({exc}); ' + fallback_note)
            return ([], f'extraction: vector query failed ({exc})')

    def extract_from_rows(self, question: str, records: List[Dict[str, Any]], question_embedding: List[float] | None=None) -> tuple[List[Dict[str, Any]], str]:
        chunk_rows: List[Dict[str, Any]] = []
        chunk_texts: List[str] = []
        for row in records:
            text = _first_chunk_text(row)
            if text is not None:
                chunk_rows.append(row)
                chunk_texts.append(text)
        if not chunk_rows:
            return (records, 'extraction: no chunk-text rows found; using full records')
        try:
            q_emb = np.asarray(question_embedding if question_embedding is not None else self._embed([question])[0], dtype=np.float64)
            q_norm = float(np.linalg.norm(q_emb))
            if q_norm == 0:
                raise ValueError('question embedding has zero norm')
            chunk_embs = np.asarray(self._embed(chunk_texts), dtype=np.float64)
            chunk_norms = np.linalg.norm(chunk_embs, axis=1)
            chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
            scores = chunk_embs @ q_emb / (chunk_norms * q_norm)
            order = np.argsort(scores)[::-1][:self._top_k_chunks]
            extracted = [chunk_rows[int(i)] for i in order]
            return (extracted, f'extraction: selected {len(extracted)} / {len(chunk_rows)} chunk rows by relevance')
        except Exception as exc:
            limit = min(len(chunk_rows), self._top_k_chunks)
            return (chunk_rows[:limit], f'extraction: row ranking failed ({exc}); using first {limit} rows')
