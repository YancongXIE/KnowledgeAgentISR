from __future__ import annotations
import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator

class RetrievalPlan(BaseModel):
    intent: str = Field(description='What the user wants from the KG')
    target_labels: List[str] = Field(default_factory=list, description='Node labels to prefer in Cypher (Publication, Element, Chunk, ...)')
    key_properties: Dict[str, str] = Field(default_factory=dict, description='Known identifiers, e.g. DOI or elementName if mentioned')
    preferred_relationships: List[str] = Field(default_factory=list, description='Relationship types that are likely relevant')

    @field_validator('preferred_relationships', mode='before')
    @classmethod
    def _coerce_preferred_relationships(cls, v: object) -> List[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return [str(v)]
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # Model sometimes returns {"pattern": ..., "description": ...}
                name = item.get('type') or item.get('name') or item.get('pattern') or ''
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
                else:
                    try:
                        out.append(json.dumps(item, ensure_ascii=False))
                    except Exception:
                        out.append(str(item))
            else:
                out.append(str(item))
        return out

    search_hint: str = Field(default='', description='Short hint for writing Cypher or vector search')

    @field_validator('search_hint', mode='before')
    @classmethod
    def _coerce_search_hint(cls, v: object) -> str:
        if v is None:
            return ''
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    @field_validator('key_properties', mode='before')
    @classmethod
    def _coerce_key_properties(cls, v: object) -> Dict[str, str]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            return {}
        out: Dict[str, str] = {}
        for key, val in v.items():
            sk = str(key)
            if val is None:
                continue
            if isinstance(val, str):
                out[sk] = val
            elif isinstance(val, dict):
                out[sk] = json.dumps(val, ensure_ascii=False)
            else:
                out[sk] = str(val)
        return out

class CypherBundle(BaseModel):
    cypher: str = Field(description='Read-only Cypher query')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters referenced as $name in the query')

class SummaryOutput(BaseModel):
    summary: str = Field(description='Condensed evidence for the answering agent')
    cited_dois: List[str] = Field(default_factory=list)

class AnalysisSelection(BaseModel):
    selected_analyses: List[str] = Field(default_factory=list, description='Ordered priority list of function_name values from the orchestrator analysis catalog (e.g. related_publications, definitions, antecedents_consequents). First item is the most important; the pipeline runs one analysis per iteration in this order.')
    objective: str = Field(default='', description='Brief objective describing what to learn this cycle')

class IntegrationOutput(BaseModel):
    stage: str = Field(description='Integration checkpoint stage name')
    merged_context: str = Field(default='', description='Updated integrated context to carry forward into subsequent steps')
    key_points: List[str] = Field(default_factory=list, description='Most important points integrated at this checkpoint')
    next_focus: str = Field(default='', description='Optional focus hint for the next retrieval/extraction/analysis cycle')

    @field_validator('merged_context', 'next_focus', mode='before')
    @classmethod
    def _null_str_to_empty_integration(cls, v: object) -> str:
        if v is None:
            return ''
        return v if isinstance(v, str) else str(v)
