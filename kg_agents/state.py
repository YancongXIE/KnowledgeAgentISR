from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    question: str
    question_embedding: Optional[List[float]] = None
    scratchpad: List[str] = field(default_factory=list)
    available_actions: List[str] = field(default_factory=list)
    selected_actions: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)
    interpretation: Optional[Dict[str, Any]] = None
    retrieval_plan: Optional[Dict[str, Any]] = None
    last_cypher: Optional[str] = None
    last_records: List[Dict[str, Any]] = field(default_factory=list)
    extracted_records: List[Dict[str, Any]] = field(default_factory=list)
    last_error: Optional[str] = None
    analysis_report: Optional[Dict[str, Any]] = None
    analysis_note: Optional[str] = None
    evidence_summary: Optional[str] = None
    summary_cited_dois: List[str] = field(default_factory=list)
    integration_memo: str = ""
    integration_trace: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    extra: Optional[str] = None
    continue_loop: bool = True
    final_answer: Optional[str] = None
    integration_note: str = ""
    pdf_focus_instruction: Optional[str] = None
    cycle_index: int = 0
    selected_analyses: List[str] = field(default_factory=list)
    selected_levels: List[str] = field(default_factory=list)
    current_objective: str = ""
    analysis_queue: List[str] = field(default_factory=list)
    analysis_step: int = 0
    stop_gathering: bool = False
    use_kg: bool = True
    research_intent: Optional[Dict[str, Any]] = None
    integration_state: Optional[Dict[str, Any]] = None
    reflection: Optional[Dict[str, Any]] = None
    vicarious: Optional[Dict[str, Any]] = None
    knowledge_package: Optional[Dict[str, Any]] = None
    recommended_analyses: List[str] = field(default_factory=list)
    interactive: bool = False
    pause_status: Optional[str] = None  # needs_clarification | awaiting_human_feedback
    resume_from: Optional[str] = None
    cycle_knowledge_packages: List[Dict[str, Any]] = field(default_factory=list)
    current_cycle_package: Optional[Dict[str, Any]] = None
    agent_memories: List[Dict[str, Any]] = field(default_factory=list)
    internalization_result: Optional[Dict[str, Any]] = None
    human_feedback: Optional[str] = None
    collaboration_prompt: Optional[str] = None

    def log(self, line: str) -> None:
        self.scratchpad.append(line)
