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


def _null_str(v: object) -> str:
    if v is None:
        return ''
    return v if isinstance(v, str) else str(v)


def _null_str_list(v: object) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            out.append(item if isinstance(item, str) else str(item))
        return out
    return [str(v)]


class ResearchIntent(BaseModel):
    """Structured research intent from human externalization (Elicitation Agent)."""

    objective: str = Field(default='', description='Inferred research objective')
    target_concepts: List[str] = Field(default_factory=list, description='Focal concepts to investigate')
    theoretical_contribution: str = Field(default='', description='Intended theoretical contribution')
    assumptions: List[str] = Field(default_factory=list, description='Inferred assumptions')
    discovery_type: str = Field(
        default='relationship',
        description='Desired discovery type: definition, relationship, proposition, research_gap, conceptual_model, etc.',
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description='Targeted clarifying questions if the prompt is underspecified (non-blocking)',
    )
    is_sufficiently_specified: bool = Field(
        default=True,
        description='False when the question is vague; pipeline still proceeds with best-effort intent',
    )
    refined_question: str = Field(default='', description='Best-effort refined research question')

    @field_validator('objective', 'theoretical_contribution', 'discovery_type', 'refined_question', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @field_validator('target_concepts', 'assumptions', 'clarifying_questions', mode='before')
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)


class ConceptualRelationship(BaseModel):
    source: str = Field(default='')
    relation: str = Field(default='')
    target: str = Field(default='')
    note: str = Field(default='')

    @field_validator('source', 'relation', 'target', 'note', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @classmethod
    def from_any(cls, value: object) -> 'ConceptualRelationship':
        if isinstance(value, ConceptualRelationship):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        text = _null_str(value).strip()
        if not text:
            return cls()
        # Parse "A -> B", "A —[rel]→ B", "A - B (note)"
        for sep in ('->[', '->', '→', '—[', '—>', '=>'):
            if sep in text:
                left, right = text.split(sep, 1)
                # strip trailing "]" from relation fragments like "[rel]→"
                right = right.lstrip('[').replace(']→', '→')
                if '→' in right or '->' in right:
                    # rare nested; treat whole as note
                    return cls(source=left.strip(), relation='related_to', target='', note=text)
                # relation may be embedded as "[rel] target"
                if ']' in right:
                    rel, tgt = right.split(']', 1)
                    return cls(source=left.strip(), relation=rel.strip('[] '), target=tgt.strip(' →->'), note='')
                return cls(source=left.strip(), relation='related_to', target=right.strip(), note='')
        return cls(source='', relation='related_to', target='', note=text)


def _coerce_relationships(v: object) -> List[ConceptualRelationship]:
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return [ConceptualRelationship.from_any(item) for item in v if item is not None]


class IntegrationState(BaseModel):
    """Salient-to-salient synthesis carried across extraction cycles."""

    stage: str = Field(default='integrate', description='Integration stage name')
    merged_context: str = Field(default='', description='Running integrated context')
    key_points: List[str] = Field(default_factory=list)
    next_focus: str = Field(default='')
    emerging_concepts: List[str] = Field(default_factory=list)
    conceptual_relationships: List[ConceptualRelationship] = Field(default_factory=list)
    theoretical_gaps: List[str] = Field(default_factory=list)
    propositions: List[str] = Field(default_factory=list)
    candidate_models: List[str] = Field(default_factory=list)
    conflicting_evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, description='0-1 confidence in the synthesis')

    @field_validator('stage', 'merged_context', 'next_focus', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @field_validator(
        'key_points',
        'emerging_concepts',
        'theoretical_gaps',
        'propositions',
        'candidate_models',
        'conflicting_evidence',
        mode='before',
    )
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)

    @field_validator('conceptual_relationships', mode='before')
    @classmethod
    def _coerce_rels(cls, v: object) -> List[ConceptualRelationship]:
        return _coerce_relationships(v)

    @field_validator('confidence', mode='before')
    @classmethod
    def _coerce_confidence(cls, v: object) -> float:
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class ReflectionDecision(BaseModel):
    """Agent-side proxy for iterative collaboration (replaces sufficiency checkpoint)."""

    sufficient: bool = Field(default=False)
    continue_loop: bool = Field(default=True, description='Whether another extraction cycle is worthwhile')
    uncertainties: List[str] = Field(default_factory=list)
    recommended_analyses: List[str] = Field(
        default_factory=list,
        description='Analysaurus function_name values that would most improve knowledge next',
    )
    follow_up_question: str = Field(default='', description='Follow-up question to explore next')
    rationale: str = Field(default='')

    @field_validator('follow_up_question', 'rationale', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @field_validator('uncertainties', 'recommended_analyses', mode='before')
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)


class VicariousReadingItem(BaseModel):
    title_or_label: str = Field(default='')
    doi: str = Field(default='')
    why_useful: str = Field(default='')
    excerpt: str = Field(default='')

    @field_validator('title_or_label', 'doi', 'why_useful', 'excerpt', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)


class VicariousLearningOutput(BaseModel):
    """Human internalization supports: qualitative / narrative readings."""

    illustrative_studies: List[VicariousReadingItem] = Field(default_factory=list)
    case_studies: List[VicariousReadingItem] = Field(default_factory=list)
    narratives: List[VicariousReadingItem] = Field(default_factory=list)
    practical_examples: List[VicariousReadingItem] = Field(default_factory=list)
    contradictory_evidence: List[VicariousReadingItem] = Field(default_factory=list)
    reading_sequence: List[str] = Field(default_factory=list)
    note: str = Field(default='')

    @field_validator('reading_sequence', mode='before')
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)

    @field_validator('note', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)


class ClaimProvenance(BaseModel):
    claim: str = Field(default='')
    source_type: str = Field(default='analysis', description='kg | chunk | analysis')
    dois: List[str] = Field(default_factory=list)
    cypher_ref: str = Field(default='')
    record_ids: List[str] = Field(default_factory=list)

    @field_validator('claim', 'source_type', 'cypher_ref', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @field_validator('dois', 'record_ids', mode='before')
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)


class KnowledgePackage(BaseModel):
    """Structured knowledge package (per-cycle intermediate or final deliverable)."""

    stage: str = Field(default='final', description='cycle_N for intermediate packages; final after vicarious')
    executive_synthesis: str = Field(default='')
    key_concepts: List[str] = Field(default_factory=list)
    conceptual_relationships: List[ConceptualRelationship] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    candidate_propositions: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    suggested_next_analyses: List[str] = Field(default_factory=list)
    recommended_qualitative_readings: List[str] = Field(default_factory=list)
    provenance: List[ClaimProvenance] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0)

    @field_validator('stage', 'executive_synthesis', mode='before')
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return _null_str(v)

    @field_validator(
        'key_concepts',
        'supporting_evidence',
        'candidate_propositions',
        'research_gaps',
        'suggested_next_analyses',
        'recommended_qualitative_readings',
        'clarifying_questions',
        mode='before',
    )
    @classmethod
    def _coerce_lists(cls, v: object) -> List[str]:
        return _null_str_list(v)

    @field_validator('conceptual_relationships', mode='before')
    @classmethod
    def _coerce_rels(cls, v: object) -> List[ConceptualRelationship]:
        return _coerce_relationships(v)

    @field_validator('confidence', mode='before')
    @classmethod
    def _coerce_confidence(cls, v: object) -> float:
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0
