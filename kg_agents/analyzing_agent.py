from __future__ import annotations
import math
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import networkx as nx
from neo4j import Driver

def _row_str(row: Dict[str, Any], key: str) -> str:
    val = row.get(key)
    if val is None:
        return ''
    return str(val).strip()

def _topic_matches(value: Any, selected: str) -> bool:
    if value is None:
        return False
    parts = [p.strip() for p in str(value).replace(' ', '').split(',') if p.strip()]
    return selected in parts

def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys([v for v in values if v]))

def _ki_entropy_index(path_counts: Dict[Any, int]) -> float:
    total = sum(path_counts.values())
    n = len(path_counts)
    if n <= 1 or total == 0:
        return 1.0
    probs = [c / total for c in path_counts.values()]
    h = -sum((p * math.log2(p) for p in probs if p > 0))
    h_max = math.log2(n)
    if h_max == 0:
        return 1.0
    return round(1.0 - h / h_max, 3)

def _ki_effective_paths_from_edges(edges: List[List[str]]) -> List[Tuple[Tuple[str, str, str], ...]]:
    if not edges:
        return []
    outgoing: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    incoming: Dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for s, direction, t in edges:
        outgoing[s].append((direction, t))
        incoming[t] += 1
        nodes.add(s)
        nodes.add(t)
    for s, _, _ in edges:
        if s not in incoming:
            incoming[s] = 0
    sources = [n for n in nodes if incoming[n] == 0]
    sinks = [n for n in nodes if not outgoing[n]]
    if not sources or not sinks:
        return []
    paths: List[Tuple[Tuple[str, str, str], ...]] = []
    visited: set[str] = set()

    def dfs(u: str, path: List[Tuple[str, str, str]]) -> None:
        visited.add(u)
        if u in sinks and path:
            paths.append(tuple(path))
        for direction, v in outgoing[u]:
            if v not in visited:
                path.append((u, direction, v))
                dfs(v, path)
                path.pop()
        visited.remove(u)
    for src in sources:
        dfs(src, [])
    return paths

def _ki_aggregate_path_counts(models: List[Dict[str, Any]], *, end_concept: Optional[str]=None, p: Optional[int]=None) -> Counter[Any]:
    path_counts: Counter[Any] = Counter()
    end_norm = (end_concept or '').strip()
    for model in models:
        seen_in_this_model: set[Any] = set()
        edges = model.get('edges', [])
        for path in _ki_effective_paths_from_edges(edges):
            last_from, last_dir, last_to = path[-1]
            if end_norm and (last_to or '').strip() != end_norm:
                continue
            if p is not None:
                k = min(len(path), p)
                if k <= 0:
                    continue
                path_fragment = path[-k:]
            else:
                path_fragment = path
            if path_fragment not in seen_in_this_model:
                path_counts[path_fragment] += 1
                seen_in_this_model.add(path_fragment)
    return path_counts

def _concept_publication_matrix(concept: str, *, driver: Driver, curation_id: str='13', limit: int=250) -> Dict[str, Any]:
    concept_value = concept.strip()
    if not concept_value:
        return {'concept': concept, 'publications': [], 'n_publications': 0, 'detail': [], 'source': 'neo4j', 'message': 'Empty concept input.'}
    query = "\n    MATCH (p:Publication)--(m:Model)--(:Relation)--(e:Element)\n    WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n      AND toLower(coalesce(e.elementName, '')) = toLower($concept)\n    RETURN DISTINCT\n      p.citation AS citation,\n      coalesce(p.DOI, '') AS doi,\n      p.uuid AS uuid,\n      e.elementName AS Concept\n    LIMIT $limit\n    "
    res, _, _ = driver.execute_query(query, curation_id=curation_id, concept=concept_value, limit=int(limit))
    detail: List[Dict[str, Any]] = []
    for rec in res:
        d = rec.data()
        detail.append({'Publication': _row_str(d, 'citation'), 'uuid': _row_str(d, 'uuid'), 'Concept': _row_str(d, 'Concept'), 'doi': _row_str(d, 'doi')})
    publications = _dedupe_keep_order([d['Publication'] for d in detail if d.get('Publication')])
    return {'concept': concept, 'publications': publications, 'n_publications': len(publications), 'detail': detail, 'source': 'neo4j'}

class AnalyzingAgent:

    def __init__(self, embed_fn: Callable[[List[str]], List[List[float]]], *, driver: Driver | None=None, curation_id: str='13', top_k: int=10, max_graph_rows: int=250, similarity_model_name: Optional[str]=None) -> None:
        self._embed = embed_fn
        self._driver = driver
        self._curation_id = curation_id
        self._top_k = top_k
        self._max_graph_rows = max_graph_rows
        self._similarity_model_name = similarity_model_name or os.environ.get('ANALYSIS_SIMILARITY_MODEL', 'bert-large-nli-stsb-mean-tokens')
        self._similarity_st_model: Any = None
        self._cached_relation_graph: Optional[Tuple[List[Dict[str, Any]], 'nx.DiGraph']] = None
        self._cached_relation_graph_curation: Optional[str] = None

    def _encode_with_sentence_transformer(self, texts: List[str]) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Definition similarity requires the `sentence-transformers` package. Install with: pip install sentence-transformers') from exc
        if self._similarity_st_model is None:
            self._similarity_st_model = SentenceTransformer(self._similarity_model_name)
        return self._similarity_st_model.encode(texts, show_progress_bar=False)

    def _filter_rows(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(r) for r in records if _topic_matches(r.get('curationID'), self._curation_id)]

    def related_publications(self, concept: str, *, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        active = driver if driver is not None else self._driver
        if not use_graph:
            return {'concept': concept, 'publications': [], 'n_publications': 0, 'detail': [], 'source': 'disabled', 'message': 'Neo4j query disabled (`use_graph=False`).'}
        if active is None:
            return {'concept': concept, 'publications': [], 'n_publications': 0, 'detail': [], 'source': 'neo4j', 'message': 'No Neo4j driver available.'}
        try:
            res = _concept_publication_matrix(concept, driver=active, curation_id=self._curation_id, limit=self._max_graph_rows)
            res['publications'] = res.get('publications', [])[:self._top_k]
            return res
        except Exception as exc:
            return {'concept': concept, 'publications': [], 'n_publications': 0, 'detail': [], 'source': 'neo4j', 'graph_error': str(exc)}

    def _definitions_for_concept(self, concept: str, *, publication_uuids: Optional[Sequence[str]], driver: Optional[Driver], use_graph: bool) -> Dict[str, Any]:
        concept_value = concept.strip()
        uuids = _dedupe_keep_order([str(u).strip() for u in publication_uuids or [] if str(u).strip()])
        active = driver if driver is not None else self._driver
        if not concept_value:
            return {'concept': concept, 'publication_uuids': uuids, 'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'neo4j', 'message': 'Empty concept input.'}
        if not use_graph:
            return {'concept': concept, 'publication_uuids': uuids, 'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'disabled', 'message': 'Neo4j query disabled (`use_graph=False`).'}
        if active is None:
            return {'concept': concept, 'publication_uuids': uuids, 'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'neo4j', 'message': 'No Neo4j driver available.'}
        query = "\n        MATCH (p:Publication)--(d:Definition)--(e:Element)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n          AND toLower(coalesce(e.elementName, '')) = toLower($concept)\n          AND (size($publication_uuids) = 0 OR p.uuid IN $publication_uuids)\n        RETURN\n          d.definition AS Definition,\n          p.citation AS Publication,\n          p.uuid AS uuid\n        LIMIT $limit\n        "
        try:
            res, _, _ = active.execute_query(query, curation_id=self._curation_id, concept=concept_value, publication_uuids=uuids, limit=int(self._max_graph_rows))
            detail: List[Dict[str, Any]] = []
            for rec in res:
                d = rec.data()
                detail.append({'Definition': _row_str(d, 'Definition'), 'Publication': _row_str(d, 'Publication'), 'uuid': _row_str(d, 'uuid')})
            return {'concept': concept, 'publication_uuids': uuids, 'definitions': [r.get('Definition', '') for r in detail if r.get('Definition')], 'n_definitions': len(detail), 'detail': detail, 'source': 'neo4j'}
        except Exception as exc:
            return {'concept': concept, 'publication_uuids': uuids, 'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'neo4j', 'graph_error': str(exc)}

    def definitions(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, concept: Optional[str]=None, publication_uuids: Optional[Sequence[str]]=None, scope_definitions_to_related_publications: bool=False, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if concept is not None:
            return self._definitions_for_concept(concept, publication_uuids=publication_uuids, driver=drv, use_graph=use_graph)
        if records is None:
            return {'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'records', 'message': 'Provide `concept=...` or `records=...`.'}
        concept_names = _dedupe_keep_order([_row_str(r, 'elementName') for r in records])
        if not concept_names:
            return {'definitions': [], 'n_definitions': 0, 'detail': [], 'source': 'records', 'message': 'No concept names found in records.'}
        merged_detail: List[Dict[str, Any]] = []
        graph_error: Optional[str] = None
        for cname in concept_names:
            pub_scope: Optional[List[str]] = None
            if scope_definitions_to_related_publications:
                one_pubs = self.related_publications(cname, driver=drv, use_graph=use_graph)
                pub_scope = _dedupe_keep_order([_row_str(d, 'uuid') for d in one_pubs.get('detail', [])])
            one_defs = self._definitions_for_concept(cname, publication_uuids=pub_scope, driver=drv, use_graph=use_graph)
            detail = one_defs.get('detail', [])
            if isinstance(detail, list):
                merged_detail.extend(detail)
            if graph_error is None and one_defs.get('graph_error'):
                graph_error = str(one_defs.get('graph_error'))
        detail_out = merged_detail[:self._max_graph_rows]
        defs_out = [str(r.get('Definition', '')).strip() for r in detail_out if r.get('Definition')]
        out: Dict[str, Any] = {'definitions': defs_out, 'n_definitions': len(merged_detail), 'detail': detail_out, 'source': 'neo4j' if use_graph else 'disabled'}
        if graph_error is not None:
            out['graph_error'] = graph_error
        return out

    def definition_similarity(self, definition_texts: Sequence[str]) -> Dict[str, Any]:
        texts = [str(x).strip() for x in definition_texts if str(x).strip()]
        if len(texts) <= 1:
            return {'n_definitions': len(texts), 'matrix': [], 'labels': [], 'skipped': True, 'message': 'Similarity requires more than one definition.'}
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            return {'n_definitions': len(texts), 'matrix': [], 'labels': [str(i + 1) for i in range(len(texts))], 'definitions': texts, 'skipped': True, 'message': 'Install scikit-learn for cosine similarity: pip install scikit-learn', 'error': str(exc), 'source': 'sklearn_unavailable'}
        try:
            embeddings = self._encode_with_sentence_transformer(texts)
            sim = cosine_similarity(embeddings)
            matrix = np.round(sim, 2).tolist()
        except RuntimeError as exc:
            return {'n_definitions': len(texts), 'matrix': [], 'labels': [str(i + 1) for i in range(len(texts))], 'definitions': texts, 'skipped': True, 'message': str(exc), 'source': 'sentence_transformers_unavailable'}
        except Exception as exc:
            return {'n_definitions': len(texts), 'matrix': [], 'labels': [str(i + 1) for i in range(len(texts))], 'definitions': texts, 'skipped': True, 'message': f'Similarity computation failed: {exc}', 'source': 'error'}
        labels = [str(i + 1) for i in range(len(texts))]
        return {'n_definitions': len(texts), 'labels': labels, 'matrix': matrix, 'definitions': texts, 'skipped': False, 'model_name': self._similarity_model_name, 'source': 'sentence_transformers'}

    def _theories_for_concept(self, concept: str, *, driver: Optional[Driver], use_graph: bool) -> Dict[str, Any]:
        concept_value = concept.strip()
        active = driver if driver is not None else self._driver
        if not concept_value:
            return {'concept': concept, 'theories': [], 'n_theories': 0, 'detail': [], 'source': 'neo4j', 'message': 'Empty concept input.'}
        if not use_graph:
            return {'concept': concept, 'theories': [], 'n_theories': 0, 'detail': [], 'source': 'disabled', 'message': 'Neo4j query disabled (`use_graph=False`).'}
        if active is None:
            return {'concept': concept, 'theories': [], 'n_theories': 0, 'detail': [], 'source': 'neo4j', 'message': 'No Neo4j driver available.'}
        query = "\n        MATCH (t:Theory)--(e:Element)\n        WHERE toLower(coalesce(e.elementName, '')) = toLower($concept)\n        RETURN DISTINCT t.theoryTitle AS Theory\n        LIMIT $limit\n        "
        try:
            res, _, _ = active.execute_query(query, concept=concept_value, limit=int(self._max_graph_rows))
            detail: List[Dict[str, Any]] = []
            for rec in res:
                d = rec.data()
                title = _row_str(d, 'Theory')
                if title:
                    detail.append({'Theory': title, 'concept': concept_value})
            theories = _dedupe_keep_order([r['Theory'] for r in detail])
            return {'concept': concept, 'theories': theories, 'n_theories': len(theories), 'detail': detail, 'source': 'neo4j'}
        except Exception as exc:
            return {'concept': concept, 'theories': [], 'n_theories': 0, 'detail': [], 'source': 'neo4j', 'graph_error': str(exc)}

    def related_theories(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, concept: Optional[str]=None, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if concept is not None:
            return self._theories_for_concept(concept, driver=drv, use_graph=use_graph)
        if records is None:
            return {'theories': [], 'n_theories': 0, 'detail': [], 'source': 'records', 'message': 'Provide `concept=...` or `records=...`.'}
        concept_names = _dedupe_keep_order([_row_str(r, 'elementName') for r in records])
        if not concept_names:
            return {'theories': [], 'n_theories': 0, 'detail': [], 'source': 'records', 'message': 'No concept names found in records.'}
        merged_detail: List[Dict[str, Any]] = []
        merged_titles: List[str] = []
        graph_error: Optional[str] = None
        for cname in concept_names:
            one = self._theories_for_concept(cname, driver=drv, use_graph=use_graph)
            dlist = one.get('detail', [])
            if isinstance(dlist, list):
                merged_detail.extend(dlist)
            merged_titles.extend([str(x) for x in one.get('theories', []) if str(x).strip()])
            if graph_error is None and one.get('graph_error'):
                graph_error = str(one.get('graph_error'))
        theories_out = _dedupe_keep_order(merged_titles)
        out: Dict[str, Any] = {'theories': theories_out, 'n_theories': len(theories_out), 'concept_names': concept_names, 'detail': merged_detail[:self._max_graph_rows], 'source': 'neo4j' if use_graph else 'disabled'}
        if graph_error is not None:
            out['graph_error'] = graph_error
        return out

    def _antecedents_consequents_for_focal(self, focal_concept: str, *, driver: Optional[Driver], use_graph: bool) -> Dict[str, Any]:
        focal = focal_concept.strip()
        active = driver if driver is not None else self._driver
        if not focal:
            return {'focal_concept': focal_concept, 'antecedents': [], 'consequents': [], 'source': 'neo4j', 'message': 'Empty focal concept.'}
        if not use_graph:
            return {'focal_concept': focal, 'antecedents': [], 'consequents': [], 'curation_id': self._curation_id, 'source': 'disabled', 'message': 'Neo4j query disabled (`use_graph=False`).'}
        if active is None:
            return {'focal_concept': focal, 'antecedents': [], 'consequents': [], 'curation_id': self._curation_id, 'source': 'neo4j', 'message': 'No Neo4j driver available.'}
        antecedent_cypher = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n        WITH DISTINCT r\n        MATCH (a:Element)<-[:HAS {role: 'antecedent'}]-(r)-[:HAS {role: 'consequent'}]->(c:Element)\n        WHERE toLower(coalesce(c.elementName, '')) = toLower($focal_concept)\n        RETURN a.elementName AS name, count(r) AS frequency\n        ORDER BY frequency DESC\n        "
        consequent_cypher = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n        WITH DISTINCT r\n        MATCH (c:Element)<-[:HAS {role: 'consequent'}]-(r)-[:HAS {role: 'antecedent'}]->(a:Element)\n        WHERE toLower(coalesce(a.elementName, '')) = toLower($focal_concept)\n        RETURN c.elementName AS name, count(r) AS frequency\n        ORDER BY frequency DESC\n        "
        params = {'focal_concept': focal, 'curation_id': self._curation_id}
        try:
            ant_res, _, _ = active.execute_query(antecedent_cypher, **params)
            cq_res, _, _ = active.execute_query(consequent_cypher, **params)
            antecedents: List[Dict[str, Any]] = []
            for rec in ant_res:
                d = rec.data()
                antecedents.append({'elementName': _row_str(d, 'name'), 'frequency': int(d.get('frequency', 0))})
            consequents: List[Dict[str, Any]] = []
            for rec in cq_res:
                d = rec.data()
                consequents.append({'elementName': _row_str(d, 'name'), 'frequency': int(d.get('frequency', 0))})
            return {'focal_concept': focal, 'antecedents': antecedents, 'consequents': consequents, 'curation_id': self._curation_id, 'source': 'neo4j'}
        except Exception as exc:
            return {'focal_concept': focal, 'antecedents': [], 'consequents': [], 'curation_id': self._curation_id, 'source': 'neo4j', 'graph_error': str(exc)}

    def antecedents_consequents(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, focal_concept: Optional[str]=None, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if focal_concept is not None:
            return self._antecedents_consequents_for_focal(focal_concept, driver=drv, use_graph=use_graph)
        if records is None:
            return {'by_focal': {}, 'source': 'records', 'message': 'Provide `records=...` or `focal_concept=...`.'}
        concept_names = _dedupe_keep_order([_row_str(r, 'elementName') for r in records])
        if not concept_names:
            return {'by_focal': {}, 'concept_names': [], 'source': 'records', 'message': 'No concept names found in records.'}
        by_focal: Dict[str, Any] = {}
        graph_error: Optional[str] = None
        for focal in concept_names:
            one = self._antecedents_consequents_for_focal(focal, driver=drv, use_graph=use_graph)
            by_focal[focal] = {'antecedents': one.get('antecedents', []), 'consequents': one.get('consequents', [])}
            if graph_error is None and one.get('graph_error'):
                graph_error = str(one.get('graph_error'))
        out: Dict[str, Any] = {'by_focal': by_focal, 'concept_names': concept_names, 'curation_id': self._curation_id, 'source': 'neo4j' if use_graph else 'disabled'}
        if graph_error is not None:
            out['graph_error'] = graph_error
        return out

    def _mediators_moderators_for_focal(self, focal_concept: str, *, driver: Optional[Driver], use_graph: bool) -> Dict[str, Any]:
        focal = focal_concept.strip()
        active = driver if driver is not None else self._driver
        if not focal:
            return {'focal_concept': focal_concept, 'moderators': [], 'mediators': [], 'source': 'neo4j', 'message': 'Empty focal concept.'}
        if not use_graph:
            return {'focal_concept': focal, 'moderators': [], 'mediators': [], 'curation_id': self._curation_id, 'source': 'disabled', 'message': 'Neo4j query disabled (`use_graph=False`).'}
        if active is None:
            return {'focal_concept': focal, 'moderators': [], 'mediators': [], 'curation_id': self._curation_id, 'source': 'neo4j', 'message': 'No Neo4j driver available.'}
        moderator_cypher = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n        WITH DISTINCT r\n        MATCH (e:Element)<-[:HAS]-(r)-[:HAS {role: 'moderator'}]->(mo:Element)\n        WHERE toLower(coalesce(e.elementName, '')) = toLower($focal_concept)\n        RETURN mo.elementName AS name, count(r) AS frequency\n        ORDER BY frequency DESC\n        "
        mediator_cypher = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n        WITH DISTINCT r\n        MATCH (e:Element)<-[:HAS]-(r)-[:HAS {role: 'mediator'}]->(mo:Element)\n        WHERE toLower(coalesce(e.elementName, '')) = toLower($focal_concept)\n        RETURN mo.elementName AS name, count(r) AS frequency\n        ORDER BY frequency DESC\n        "
        params = {'focal_concept': focal, 'curation_id': self._curation_id}
        try:
            mod_res, _, _ = active.execute_query(moderator_cypher, **params)
            med_res, _, _ = active.execute_query(mediator_cypher, **params)
            moderators: List[Dict[str, Any]] = []
            for rec in mod_res:
                d = rec.data()
                moderators.append({'elementName': _row_str(d, 'name'), 'frequency': int(d.get('frequency', 0))})
            mediators: List[Dict[str, Any]] = []
            for rec in med_res:
                d = rec.data()
                mediators.append({'elementName': _row_str(d, 'name'), 'frequency': int(d.get('frequency', 0))})
            return {'focal_concept': focal, 'moderators': moderators, 'mediators': mediators, 'curation_id': self._curation_id, 'source': 'neo4j'}
        except Exception as exc:
            return {'focal_concept': focal, 'moderators': [], 'mediators': [], 'curation_id': self._curation_id, 'source': 'neo4j', 'graph_error': str(exc)}

    def mediators_moderators(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, focal_concept: Optional[str]=None, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if focal_concept is not None:
            return self._mediators_moderators_for_focal(focal_concept, driver=drv, use_graph=use_graph)
        if records is None:
            return {'by_focal': {}, 'source': 'records', 'message': 'Provide `records=...` or `focal_concept=...`.'}
        concept_names = _dedupe_keep_order([_row_str(r, 'elementName') for r in records])
        if not concept_names:
            return {'by_focal': {}, 'concept_names': [], 'source': 'records', 'message': 'No concept names found in records.'}
        by_focal: Dict[str, Any] = {}
        graph_error: Optional[str] = None
        for focal in concept_names:
            one = self._mediators_moderators_for_focal(focal, driver=drv, use_graph=use_graph)
            by_focal[focal] = {'moderators': one.get('moderators', []), 'mediators': one.get('mediators', [])}
            if graph_error is None and one.get('graph_error'):
                graph_error = str(one.get('graph_error'))
        out: Dict[str, Any] = {'by_focal': by_focal, 'concept_names': concept_names, 'curation_id': self._curation_id, 'source': 'neo4j' if use_graph else 'disabled'}
        if graph_error is not None:
            out['graph_error'] = graph_error
        return out

    def _centrality_disabled(self, reason: str) -> Dict[str, Any]:
        return {'source': 'disabled', 'curation_id': self._curation_id, 'message': reason, 'ranking': [], 'concept_measures': []}

    @staticmethod
    def _centrality_edges_from_rows(rows: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for row in rows:
            ant = _row_str(row, 'antecedent')
            cons = _row_str(row, 'consequent')
            med = _row_str(row, 'mediator')
            if not ant or not cons:
                continue
            if med:
                edges.append((ant, med))
                edges.append((med, cons))
            else:
                edges.append((ant, cons))
        return edges
    _RELATION_GRAPH_CYPHER = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n          AND coalesce(r.type, '') <> 'moderating'\n        WITH DISTINCT r\n        MATCH (e2:Element)<-[c:HAS {role: 'consequent'}]-(r)-[a:HAS {role: 'antecedent'}]->(e1:Element)\n        OPTIONAL MATCH (r)-[med:HAS {role: 'mediator'}]->(mediator:Element)\n        RETURN e1.elementName AS antecedent,\n               mediator.elementName AS mediator,\n               e2.elementName AS consequent\n        "
    _ASSOCIATION_TRANSACTIONS_CYPHER = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n          AND coalesce(r.type, '') <> 'moderating'\n        MATCH (e:Element)<-[:HAS]-(r)\n        RETURN DISTINCT coalesce(m.uuid, toString(id(m))) AS model_id,\n               coalesce(e.elementName, '') AS concept\n        "
    _KI_CAUSAL_EDGES_CYPHER = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n          AND coalesce(r.type, '') = 'causal'\n        MATCH (a:Element)<-[:HAS {role: 'antecedent'}]-(r)-[:HAS {role: 'consequent'}]->(c:Element)\n        RETURN coalesce(m.uuid, toString(id(m))) AS model_id,\n               coalesce(a.elementName, '') AS ant,\n               'link' AS direction,\n               coalesce(c.elementName, '') AS cons\n        "
    _KI_CONSEQUENT_FREQ_CYPHER = "\n        MATCH (p:Publication)--(m:Model)-[:DEPICTS]->(r:Relation)\n        WHERE ($curation_id = '' OR $curation_id IN split(replace(coalesce(p.curationID, ''), ' ', ''), ','))\n          AND coalesce(r.type, '') = 'causal'\n        MATCH (c:Element)<-[rc:HAS {role: 'consequent'}]-(r)\n        RETURN coalesce(c.elementName, '') AS consequent, count(DISTINCT rc) AS frequency\n        ORDER BY frequency DESC\n        "

    def _load_relation_graph(self, driver: Driver) -> Tuple[List[Dict[str, Any]], nx.DiGraph]:
        if (self._cached_relation_graph is not None
                and self._cached_relation_graph_curation == self._curation_id):
            return self._cached_relation_graph
        res, _, _ = driver.execute_query(self._RELATION_GRAPH_CYPHER, curation_id=self._curation_id)
        raw_rows: List[Dict[str, Any]] = []
        for rec in res:
            raw_rows.append(rec.data())
        edges = self._centrality_edges_from_rows(raw_rows)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        self._cached_relation_graph = (raw_rows, G)
        self._cached_relation_graph_curation = self._curation_id
        return (raw_rows, G)

    def invalidate_graph_cache(self) -> None:
        """Call this to force a fresh graph load on the next analysis."""
        self._cached_relation_graph = None
        self._cached_relation_graph_curation = None

    @staticmethod
    def _cutpoints(G: nx.DiGraph) -> List[Tuple[str, int]]:
        if G.number_of_nodes() == 0:
            return []
        base = nx.number_weakly_connected_components(G)
        deltas: List[Tuple[str, int]] = []
        for v in list(G.nodes()):
            H = G.copy()
            H.remove_node(v)
            new_c = nx.number_weakly_connected_components(H)
            delta = new_c - base
            if delta > 0:
                deltas.append((str(v), int(delta)))
        deltas.sort(key=lambda x: (-x[1], x[0]))
        return deltas

    @staticmethod
    def _periphery_table(G: nx.DiGraph) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if G.number_of_nodes() == 0:
            return ([], None)
        U = G.to_undirected()
        n_all = U.number_of_nodes()
        note: Optional[str] = None
        if not nx.is_connected(U):
            largest = max(nx.connected_components(U), key=len)
            U = U.subgraph(largest).copy()
            note = f'Eccentricity is on the largest undirected connected component ({U.number_of_nodes()} of {n_all} nodes).'
        if U.number_of_nodes() == 0:
            return ([], note)
        ecc_map = nx.eccentricity(U)
        max_e = max(ecc_map.values()) if ecc_map else 0.0
        rows: List[Dict[str, Any]] = []
        for n, e in ecc_map.items():
            e_f = float(e)
            per = e_f / max_e if max_e > 0 else 0.0
            rows.append({'Concept': str(n), 'Eccentricity': e_f, 'Periphery': per})
        rows.sort(key=lambda x: (-x['Eccentricity'], x['Concept']))
        return (rows, note)

    @staticmethod
    def _finite_float(x: Any) -> bool:
        if not isinstance(x, (int, float)):
            return False
        if isinstance(x, float):
            return not (math.isnan(x) or math.isinf(x))
        return True

    @staticmethod
    def _json_float(x: Any) -> Optional[float]:
        if not isinstance(x, (int, float)):
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)

    @staticmethod
    def _rank_structural_hole_values(values: Dict[Any, float], *, n: int, reverse: bool, value_key: str) -> List[Dict[str, Any]]:
        items = [(str(k), float(v)) for k, v in values.items() if AnalyzingAgent._finite_float(v)]
        items.sort(key=lambda t: (-t[1] if reverse else t[1], t[0]))
        out: List[Dict[str, Any]] = []
        for name, val in items[:max(0, n)]:
            row: Dict[str, Any] = {'Concept': name, value_key: val}
            out.append(row)
        return out

    @staticmethod
    def _apriori_association_rules(transactions: List[List[str]], *, min_support: float, min_confidence: float, max_len: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        try:
            import pandas as pd
            from mlxtend.frequent_patterns import apriori as mlx_apriori
            from mlxtend.frequent_patterns import association_rules as mlx_arules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError as exc:
            return ([], f'Optional dependency missing for association rules: {exc}')
        if not transactions:
            return ([], None)
        n_tx = len(transactions)
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        if df.shape[1] == 0:
            return ([], None)
        frequent_itemsets = mlx_apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
        if frequent_itemsets.empty:
            return ([], None)
        rules_df = mlx_arules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        if rules_df.empty:
            return ([], None)
        out: List[Dict[str, Any]] = []
        for _, row in rules_df.iterrows():
            lhs = row['antecedents']
            rhs = row['consequents']
            c1 = ', '.join(sorted((str(x) for x in lhs)))
            c2 = ', '.join(sorted((str(x) for x in rhs)))
            supp = float(row['support'])
            conf = float(row['confidence'])
            lift_v = float(row['lift'])
            if not math.isfinite(lift_v):
                lift_v = None
            cov = float(row['antecedent support'])
            cnt = int(round(supp * n_tx))
            out.append({'concept1': c1, 'concept2': c2, 'support': supp, 'confidence': conf, 'coverage': cov, 'lift': lift_v, 'count': cnt})
        return (out, None)

    def _compute_centrality_bundle(self, records: Optional[Sequence[Dict[str, Any]]], *, driver: Optional[Driver], use_graph: bool) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if not use_graph:
            return {'indegree': self._centrality_disabled('Neo4j query disabled (`use_graph=False`).'), 'outdegree': self._centrality_disabled('Neo4j query disabled (`use_graph=False`).'), 'betweenness': self._centrality_disabled('Neo4j query disabled (`use_graph=False`).')}
        if drv is None:
            return {'indegree': self._centrality_disabled('No Neo4j driver available.'), 'outdegree': self._centrality_disabled('No Neo4j driver available.'), 'betweenness': self._centrality_disabled('No Neo4j driver available.')}
        try:
            raw_rows, G = self._load_relation_graph(drv)
        except Exception as exc:
            err: Dict[str, Any] = {'source': 'neo4j', 'curation_id': self._curation_id, 'graph_error': str(exc), 'ranking': [], 'concept_measures': []}
            return {'indegree': {**err}, 'outdegree': {**err}, 'betweenness': {**err}}
        indegree = dict(G.in_degree())
        outdegree = dict(G.out_degree())
        between_raw = nx.betweenness_centrality(G, normalized=False)
        n_top = int(self._top_k)
        indegree_rank = [{'Concept': n, 'Indegree': int(d)} for n, d in sorted(indegree.items(), key=lambda x: (-x[1], x[0]))[:n_top]]
        outdegree_rank = [{'Concept': n, 'Outdegree': int(d)} for n, d in sorted(outdegree.items(), key=lambda x: (-x[1], x[0]))[:n_top]]
        between_rank = [{'Concept': n, 'Betweenness': round(float(between_raw[n]), 1)} for n in sorted(between_raw.keys(), key=lambda x: (-between_raw[x], x))[:n_top]]

        def _node_ci(name: str) -> Optional[str]:
            target = name.strip().lower()
            if not target:
                return None
            for n in G.nodes():
                if str(n).strip().lower() == target:
                    return str(n)
            return None
        concept_measures: List[Dict[str, Any]] = []
        seen_ci: set[str] = set()
        if records:
            for r in records:
                el = _row_str(r, 'elementName')
                ck = el.strip().lower()
                if not ck or ck in seen_ci:
                    continue
                node = _node_ci(el)
                if node is None:
                    continue
                seen_ci.add(ck)
                concept_measures.append({'Concept': node, 'Indegree': int(G.in_degree(node)), 'Outdegree': int(G.out_degree(node)), 'Betweenness': round(float(between_raw.get(node, 0.0)), 2)})
        base_meta = {'source': 'neo4j', 'curation_id': self._curation_id, 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges(), 'concept_measures': concept_measures}
        return {'indegree': {**base_meta, 'ranking': indegree_rank}, 'outdegree': {**base_meta, 'ranking': outdegree_rank}, 'betweenness': {**base_meta, 'ranking': between_rank}}

    def centrality_bundle(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        return self._compute_centrality_bundle(records, driver=driver, use_graph=use_graph)

    def cutpoints(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        if not use_graph:
            return {'source': 'disabled', 'curation_id': self._curation_id, 'message': 'Neo4j query disabled (`use_graph=False`).', 'n_weak_components_base': None, 'ranking': []}
        if drv is None:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': 'No Neo4j driver available.', 'n_weak_components_base': None, 'ranking': []}
        try:
            _raw, G = self._load_relation_graph(drv)
        except Exception as exc:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'graph_error': str(exc), 'n_weak_components_base': None, 'ranking': []}
        base = nx.number_weakly_connected_components(G)
        deltas = self._cutpoints(G)
        n_top = int(self._top_k)
        ranking = [{'Concept': name, 'Components_number_change': d} for name, d in deltas[:n_top]]
        return {'source': 'neo4j', 'curation_id': self._curation_id, 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges(), 'n_weak_components_base': int(base), 'ranking': ranking}

    def periphery(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        n_top = int(self._top_k)
        if not use_graph:
            return {'source': 'disabled', 'curation_id': self._curation_id, 'message': 'Neo4j query disabled (`use_graph=False`).', 'note': None, 'n_nodes': None, 'n_edges': None, 'highest_eccentricity': [], 'lowest_eccentricity': [], 'concept_measures': []}
        if drv is None:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': 'No Neo4j driver available.', 'note': None, 'n_nodes': None, 'n_edges': None, 'highest_eccentricity': [], 'lowest_eccentricity': [], 'concept_measures': []}
        try:
            _raw, G = self._load_relation_graph(drv)
        except Exception as exc:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'graph_error': str(exc), 'note': None, 'n_nodes': None, 'n_edges': None, 'highest_eccentricity': [], 'lowest_eccentricity': [], 'concept_measures': []}
        table, ecc_note = self._periphery_table(G)
        lowest_sorted = sorted(table, key=lambda x: (x['Eccentricity'], str(x['Concept'])))
        high = [{'Eccentricity': r['Eccentricity'], 'Concept': r['Concept'], 'Periphery': r['Periphery']} for r in table[:n_top]]
        low = [{'Eccentricity': r['Eccentricity'], 'Concept': r['Concept'], 'Periphery': r['Periphery']} for r in lowest_sorted[:n_top]]

        def _node_ci(name: str) -> Optional[str]:
            target = name.strip().lower()
            if not target:
                return None
            for n in G.nodes():
                if str(n).strip().lower() == target:
                    return str(n)
            return None
        in_table = {str(r['Concept']).strip().lower(): r for r in table}
        concept_measures: List[Dict[str, Any]] = []
        seen_ci: set[str] = set()
        if records:
            for r in records:
                el = _row_str(r, 'elementName')
                ck = el.strip().lower()
                if not ck or ck in seen_ci:
                    continue
                node = _node_ci(el)
                if node is None:
                    continue
                seen_ci.add(ck)
                row = in_table.get(node.strip().lower())
                if row is None:
                    continue
                concept_measures.append({'Concept': node, 'Periphery': round(float(row['Periphery']), 2)})
        return {'source': 'neo4j', 'curation_id': self._curation_id, 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges(), 'note': ecc_note, 'highest_eccentricity': high, 'lowest_eccentricity': low, 'concept_measures': concept_measures}

    def structural_hole_measures(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True, report_measure: Optional[str]=None, report_order: str='desc') -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        n_top = int(self._top_k)
        empty = {'source': 'disabled', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': None, 'n_nodes': None, 'n_edges': None, 'highest_constraint': [], 'lowest_constraint': [], 'highest_effective_size': [], 'lowest_effective_size': [], 'concept_measures': [], 'report': []}
        if not use_graph:
            empty['source'] = 'disabled'
            empty['message'] = 'Neo4j query disabled (`use_graph=False`).'
            return empty
        if drv is None:
            empty['source'] = 'neo4j'
            empty['message'] = 'No Neo4j driver available.'
            return empty
        try:
            _raw, G = self._load_relation_graph(drv)
        except Exception as exc:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': str(exc), 'note': None, 'n_nodes': None, 'n_edges': None, 'highest_constraint': [], 'lowest_constraint': [], 'highest_effective_size': [], 'lowest_effective_size': [], 'concept_measures': [], 'report': []}
        constraint_raw = nx.constraint(G)
        eff_raw = nx.effective_size(G)
        highest_constraint = self._rank_structural_hole_values(constraint_raw, n=n_top, reverse=True, value_key='Constraint')
        lowest_constraint = self._rank_structural_hole_values(constraint_raw, n=n_top, reverse=False, value_key='Constraint')
        highest_effective_size = self._rank_structural_hole_values(eff_raw, n=n_top, reverse=True, value_key='Effsize')
        lowest_effective_size = self._rank_structural_hole_values(eff_raw, n=n_top, reverse=False, value_key='Effsize')
        report: List[Dict[str, Any]] = []
        order_desc = str(report_order).lower() != 'asc'
        rm = (report_measure or '').strip().lower()
        if rm == 'constraint':
            report = self._rank_structural_hole_values(constraint_raw, n=n_top, reverse=order_desc, value_key='Constraint')
        elif rm in ('effective_size', 'effsize', 'effective size'):
            report = self._rank_structural_hole_values(eff_raw, n=n_top, reverse=order_desc, value_key='Effsize')

        def _node_ci(name: str) -> Optional[str]:
            target = name.strip().lower()
            if not target:
                return None
            for n in G.nodes():
                if str(n).strip().lower() == target:
                    return str(n)
            return None
        concept_measures: List[Dict[str, Any]] = []
        seen_ci: set[str] = set()
        if records:
            for r in records:
                el = _row_str(r, 'elementName')
                ck = el.strip().lower()
                if not ck or ck in seen_ci:
                    continue
                node = _node_ci(el)
                if node is None:
                    continue
                seen_ci.add(ck)
                c_val = self._json_float(constraint_raw.get(node))
                e_val = self._json_float(eff_raw.get(node))
                row_out: Dict[str, Any] = {'Concept': node}
                if c_val is not None:
                    row_out['Constraint'] = round(c_val, 2)
                else:
                    row_out['Constraint'] = None
                if e_val is not None:
                    row_out['Effsize'] = round(e_val, 2)
                else:
                    row_out['Effsize'] = None
                concept_measures.append(row_out)
        return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': 'Constraint and effective size from NetworkX (Burt); isolated or undefined nodes are excluded from rankings.', 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges(), 'highest_constraint': highest_constraint, 'lowest_constraint': lowest_constraint, 'highest_effective_size': highest_effective_size, 'lowest_effective_size': lowest_effective_size, 'concept_measures': concept_measures, 'report': report}

    def association_rules(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True, min_support: float=0.04, min_confidence: float=0.8, max_len: int=4) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        base_empty: Dict[str, Any] = {'source': 'disabled', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': None, 'n_transactions': 0, 'n_rules': 0, 'min_support': min_support, 'min_confidence': min_confidence, 'max_len': max_len, 'rules': []}
        if not use_graph:
            base_empty['message'] = 'Neo4j query disabled (`use_graph=False`).'
            return base_empty
        if drv is None:
            base_empty['source'] = 'neo4j'
            base_empty['message'] = 'No Neo4j driver available.'
            return base_empty
        try:
            res, _, _ = drv.execute_query(self._ASSOCIATION_TRANSACTIONS_CYPHER, curation_id=self._curation_id)
        except Exception as exc:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': str(exc), 'note': None, 'n_transactions': 0, 'n_rules': 0, 'min_support': min_support, 'min_confidence': min_confidence, 'max_len': max_len, 'rules': []}
        by_model: Dict[str, set[str]] = defaultdict(set)
        for rec in res:
            d = rec.data()
            mid = _row_str(d, 'model_id')
            concept = _row_str(d, 'concept').strip()
            if not mid or not concept:
                continue
            by_model[mid].add(concept)
        filter_terms: set[str] = set()
        if records:
            for r in records:
                el = _row_str(r, 'elementName').strip().lower()
                if el:
                    filter_terms.add(el)
        if filter_terms:
            filtered: Dict[str, set[str]] = {}
            for mid, items in by_model.items():
                lowered = {x.strip().lower() for x in items}
                if filter_terms & lowered:
                    filtered[mid] = items
            by_model = filtered
        transactions = [sorted(list(s)) for s in by_model.values() if s]
        n_tx = len(transactions)
        rules, dep_msg = self._apriori_association_rules(transactions, min_support=min_support, min_confidence=min_confidence, max_len=max_len)
        note_parts: List[str] = []
        if dep_msg:
            note_parts.append(dep_msg)
        if filter_terms:
            note_parts.append('Transactions restricted to models that include at least one input elementName.')
        if n_tx == 0:
            note_parts.append('No transactions after query/filter; cannot mine rules.')
        return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': '; '.join(note_parts) if note_parts else None, 'n_transactions': n_tx, 'n_rules': len(rules), 'min_support': min_support, 'min_confidence': min_confidence, 'max_len': max_len, 'rules': rules}

    def knowledge_index(self, records: Optional[Sequence[Dict[str, Any]]]=None, *, driver: Optional[Driver]=None, use_graph: bool=True, min_consequent_frequency: int=3, max_degrees: int=3, max_consequents: int=15) -> Dict[str, Any]:
        drv = driver if driver is not None else self._driver
        empty: Dict[str, Any] = {'source': 'disabled', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': None, 'n_models': 0, 'consequent_distribution_ki': None, 'by_consequent': {}, 'frequent_consequents_used': [], 'min_consequent_frequency': min_consequent_frequency, 'max_degrees': max_degrees}
        if not use_graph:
            empty['message'] = 'Neo4j query disabled (`use_graph=False`).'
            return empty
        if drv is None:
            empty['source'] = 'neo4j'
            empty['message'] = 'No Neo4j driver available.'
            return empty
        try:
            res_edges, _, _ = drv.execute_query(self._KI_CAUSAL_EDGES_CYPHER, curation_id=self._curation_id)
            res_freq, _, _ = drv.execute_query(self._KI_CONSEQUENT_FREQ_CYPHER, curation_id=self._curation_id)
        except Exception as exc:
            return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': str(exc), 'note': None, 'n_models': 0, 'consequent_distribution_ki': None, 'by_consequent': {}, 'frequent_consequents_used': [], 'min_consequent_frequency': min_consequent_frequency, 'max_degrees': max_degrees}
        by_model: Dict[str, List[List[str]]] = defaultdict(list)
        for rec in res_edges:
            d = rec.data()
            mid = _row_str(d, 'model_id')
            ant = _row_str(d, 'ant').strip()
            dire = 'link'
            cons = _row_str(d, 'cons').strip()
            if not mid or not ant or (not cons):
                continue
            by_model[mid].append([ant, dire, cons])
        models: List[Dict[str, Any]] = [{'edges': e} for e in by_model.values() if e]
        n_models = len(models)
        freq_rows: List[Tuple[str, int]] = []
        for rec in res_freq:
            dd = rec.data()
            name = _row_str(dd, 'consequent').strip()
            fr = dd.get('frequency')
            try:
                nfr = int(fr) if fr is not None else 0
            except (TypeError, ValueError):
                nfr = 0
            if name and nfr > 0:
                freq_rows.append((name, nfr))
        dist_counter: Counter[str] = Counter()
        for name, nfr in freq_rows:
            dist_counter[name] += nfr
        consequent_distribution_ki: Optional[float] = None
        if dist_counter:
            consequent_distribution_ki = _ki_entropy_index(dict(dist_counter))
        min_f = max(1, int(min_consequent_frequency))
        frequent = [n for n, f in freq_rows if f >= min_f][:max(1, int(max_consequents))]
        filter_terms: set[str] = set()
        if records:
            for r in records:
                el = _row_str(r, 'elementName').strip().lower()
                if el:
                    filter_terms.add(el)
        if filter_terms:
            frequent = [c for c in frequent if c.strip().lower() in filter_terms]
            if not frequent:
                frequent = [n for n, f in freq_rows if f >= min_f][:max(1, int(max_consequents))]
        md = max(1, min(int(max_degrees), 3))
        by_cons: Dict[str, Dict[str, float]] = {}
        for cons_name in frequent:
            row_deg: Dict[str, float] = {}
            for p in range(1, md + 1):
                pc = _ki_aggregate_path_counts(models, end_concept=cons_name, p=p)
                row_deg[f'degree_{p}'] = _ki_entropy_index(dict(pc))
            by_cons[cons_name] = row_deg
        note = "KI/pKIZscore.ipynb: path-fragment entropy on causal models under curation; consequent_distribution_ki uses marginal consequent frequencies; edge middle label is 'link' (no Relation.direction in database)."
        return {'source': 'neo4j', 'curation_id': self._curation_id, 'message': None, 'graph_error': None, 'note': note, 'n_models': n_models, 'consequent_distribution_ki': consequent_distribution_ki, 'by_consequent': by_cons, 'frequent_consequents_used': frequent, 'min_consequent_frequency': min_f, 'max_degrees': md}

    def analyze(self, records: List[Dict[str, Any]], *, driver: Driver | None=None, enrich_from_graph: bool=True) -> tuple[Dict[str, Any], str]:
        return self.analyze_selected(records, selected_analyses=None, driver=driver, enrich_from_graph=enrich_from_graph)

    def analyze_selected(self, records: List[Dict[str, Any]], *, selected_analyses: Optional[Sequence[str]]=None, driver: Driver | None=None, enrich_from_graph: bool=True) -> tuple[Dict[str, Any], str]:
        rows = self._filter_rows(records)
        drv = driver if driver is not None else self._driver
        selected = [str(x).strip() for x in selected_analyses or [] if str(x).strip()]
        selected_set = set(selected)
        run_all = len(selected_set) == 0
        concept_names = _dedupe_keep_order([_row_str(r, 'elementName') for r in rows])
        l1: Dict[str, Any] = {}
        l2: Dict[str, Any] = {}
        defs_bundle: Optional[Dict[str, Any]] = None
        if run_all or 'related_publications' in selected_set:
            merged_detail: List[Dict[str, Any]] = []
            merged_publications: List[str] = []
            graph_error: Optional[str] = None
            for concept in concept_names:
                one = self.related_publications(concept, driver=drv, use_graph=enrich_from_graph)
                merged_publications.extend(one.get('publications', []))
                detail = one.get('detail', [])
                if isinstance(detail, list):
                    merged_detail.extend(detail)
                if graph_error is None and one.get('graph_error'):
                    graph_error = str(one.get('graph_error'))
            rp: Dict[str, Any] = {'publications': _dedupe_keep_order(merged_publications)[:self._top_k], 'n_publications': len(_dedupe_keep_order(merged_publications)), 'detail': merged_detail[:self._max_graph_rows], 'concept_names': concept_names, 'source': 'neo4j' if enrich_from_graph else 'disabled'}
            if graph_error is not None:
                rp['graph_error'] = graph_error
            l1['related_publications'] = rp
        if run_all or 'definitions' in selected_set or 'definition_similarity' in selected_set:
            defs_bundle = self.definitions(records=rows, driver=drv, use_graph=enrich_from_graph)
            l1['definitions'] = defs_bundle
        if run_all or 'definition_similarity' in selected_set:
            if defs_bundle is None:
                defs_bundle = self.definitions(records=rows, driver=drv, use_graph=enrich_from_graph)
                l1['definitions'] = defs_bundle
            l1['definition_similarity'] = self.definition_similarity(defs_bundle.get('definitions', []))
        if run_all or 'related_theories' in selected_set:
            l1['related_theories'] = self.related_theories(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'antecedents_consequents' in selected_set:
            l2['antecedents_consequents'] = self.antecedents_consequents(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'mediators_moderators' in selected_set:
            l2['mediators_moderators'] = self.mediators_moderators(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or any((x in selected_set for x in ('indegreecentrality', 'outdegreecentrality', 'betweennesscentrality'))):
            centrality = self.centrality_bundle(rows, driver=drv, use_graph=enrich_from_graph)
            if run_all or 'indegreecentrality' in selected_set:
                l2['indegreecentrality'] = centrality['indegree']
            if run_all or 'outdegreecentrality' in selected_set:
                l2['outdegreecentrality'] = centrality['outdegree']
            if run_all or 'betweennesscentrality' in selected_set:
                l2['betweennesscentrality'] = centrality['betweenness']
        if run_all or 'cutpoints' in selected_set:
            l2['cutpoints'] = self.cutpoints(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'periphery' in selected_set:
            l2['periphery'] = self.periphery(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'structural_hole_measures' in selected_set:
            l2['structural_hole_measures'] = self.structural_hole_measures(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'association_rules' in selected_set:
            l2['association_rules'] = self.association_rules(records=rows, driver=drv, use_graph=enrich_from_graph)
        if run_all or 'knowledge_index' in selected_set:
            l2['knowledge_index'] = self.knowledge_index(records=rows, driver=drv, use_graph=enrich_from_graph)
        l1['n_related_publications'] = int((l1.get('related_publications') or {}).get('n_publications', 0) if isinstance(l1.get('related_publications'), dict) else 0)
        l1['definitions_found'] = int((l1.get('definitions') or {}).get('n_definitions', 0) if isinstance(l1.get('definitions'), dict) else 0)
        l1['n_related_theories'] = int((l1.get('related_theories') or {}).get('n_theories', 0) if isinstance(l1.get('related_theories'), dict) else 0)
        report = {'level_1_concept_extraction': l1, 'level_2_relationship_mining': l2}
        notes = f'Applied curation filter curation={self._curation_id}; rows_in={len(records)} rows_used={len(rows)}; selected={(selected if selected else ['ALL'])}.'
        return (report, notes)
