from __future__ import annotations
import json
from typing import Any, Dict, List
from neo4j import Driver
from .cypher_safety import validate_read_only_cypher
from .llm_json import complete_json
from .models import CypherBundle, RetrievalPlan
from .schema_agent import SchemaAgent

class GraphQueryAgent:

    def __init__(self, client: Any, schema: SchemaAgent, model: str='gpt-5.2') -> None:
        self._client = client
        self._schema = schema
        self._model = model

    def propose_query(self, question: str, plan: RetrievalPlan, question_embedding: List[float] | None, extra_context: str | None=None) -> CypherBundle:
        emb_note = ''
        if question_embedding is not None:
            emb_note = f"A question embedding is already computed as parameter $embedding (length {len(question_embedding)}). You may use CALL db.index.vector.queryNodes('chunkEmb', $candidate_pool, $embedding) when vector search over Chunk is appropriate."
        system = 'You are the graph query agent. Write a single READ-ONLY Cypher query for Neo4j. Use parameters ($name) for all literals. No CREATE/MERGE/DELETE/SET.\nPrefer targeted MATCH patterns over scanning the whole graph. If the user needs textual evidence, combine graph patterns with vector search when useful.\n\n' + self._schema.prompt_fragment() + '\n\n' + emb_note
        user = json.dumps({'question': question, 'retrieval_plan': plan.model_dump(), 'extra_context': extra_context}, ensure_ascii=False, indent=2)
        try:
            return complete_json(self._client, model=self._model, system=system, user=user, schema_model=CypherBundle)
        except Exception:
            if question_embedding is not None:
                return CypherBundle(cypher="CALL db.index.vector.queryNodes('chunkEmb', $candidate_pool, $embedding) YIELD node, score OPTIONAL MATCH (node)<-[:HAS_CHUNK]-(p:Publication) RETURN node.text AS text, score AS score, p.DOI AS doi ORDER BY score DESC LIMIT $k", parameters={'candidate_pool': 100, 'embedding': question_embedding, 'k': 10}, rationale='Fallback query: vector retrieval over chunks.')
            return CypherBundle(cypher='MATCH (p:Publication) RETURN p.DOI AS doi LIMIT $k', parameters={'k': 10}, rationale='Fallback query: lightweight publication sample.')

    def execute_safe(self, driver: Driver, bundle: CypherBundle) -> tuple[List[Dict[str, Any]], str | None]:
        ok, reason = validate_read_only_cypher(bundle.cypher)
        if not ok:
            return ([], reason)
        try:
            records, _, _ = driver.execute_query(bundle.cypher, **bundle.parameters)
            return ([dict(r) for r in records], None)
        except Exception as exc:
            return ([], str(exc))
