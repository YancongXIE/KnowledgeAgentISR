from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from .orchestrator import AzureOpenAIClient, KGMultiAgentOrchestrator, OrchestratorResult
from .pdf_context import summarize_sources
from .state import AgentState


def load_environment() -> None:
    root = Path(__file__).resolve().parent
    for env_path in (root / ".env", root.parent / ".env"):
        if env_path.exists():
            load_dotenv(env_path)
            return
    load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def build_embed_fn() -> Callable[[List[str]], List[List[float]]]:
    """Use sentence-transformers locally so the pipeline can reuse one model instance."""
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    def embed_fn(texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    return embed_fn


@dataclass
class AgentRuntime:
    driver: Driver
    orchestrator: KGMultiAgentOrchestrator
    embed_fn: Callable[[List[str]], List[List[float]]]

    def ask(
        self,
        question: str,
        *,
        use_kg: Optional[bool] = None,
        compare_baseline: bool = False,
        document_records: Optional[List[Dict[str, Any]]] = None,
    ) -> OrchestratorResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")
        question_embedding = self.embed_fn([cleaned_question])[0]
        if document_records:
            return self._answer_from_documents(
                cleaned_question,
                question_embedding=question_embedding,
                document_records=document_records,
            )
        return self.orchestrator.run(
            cleaned_question,
            question_embedding=question_embedding,
            use_kg=use_kg,
            compare_baseline=compare_baseline,
        )

    def close(self) -> None:
        self.driver.close()

    def _answer_from_documents(
        self,
        question: str,
        *,
        question_embedding: List[float],
        document_records: List[Dict[str, Any]],
    ) -> OrchestratorResult:
        if not document_records:
            raise ValueError("No document content was available after parsing the uploaded PDFs.")

        summary_out = self.orchestrator.summarizing.summarize(
            document_records,
            question=question,
            max_records_in_prompt=20,
            max_chars_per_record=2200,
        )
        source_names = summarize_sources(document_records) or "uploaded PDF files"
        analysis_note = f"Answer grounded in uploaded PDF evidence from: {source_names}."
        integrated = self.orchestrator.integration.integrate(
            stage="uploaded_pdf_context",
            question=question,
            pdf_summary=summary_out.summary,
            integration_memo="",
            kg_records=[],
            analysis_report=None,
            analysis_note=analysis_note,
            max_kg_rows=0,
        )
        final_answer = self.orchestrator.integration.compose_final_answer(
            question=question,
            pdf_summary=summary_out.summary,
            integration_memo=integrated.merged_context,
            kg_records=[],
            analysis_report=None,
            analysis_note=analysis_note,
            max_kg_rows=0,
        )
        state = AgentState(
            question=question,
            question_embedding=question_embedding,
            extracted_records=list(document_records),
            analysis_note=analysis_note,
            evidence_summary=summary_out.summary,
            summary_cited_dois=list(summary_out.cited_dois),
            integration_memo=integrated.merged_context,
            integration_trace=[
                {
                    "stage": "uploaded_pdf_context",
                    "source_names": source_names,
                    "excerpt_count": len(document_records),
                }
            ],
            iteration=1,
            final_answer=final_answer,
            use_kg=False,
        )
        return OrchestratorResult(answer=final_answer, state=state, iterations_used=1)


def create_runtime(
    *,
    use_kg: bool = True,
    log_progress: bool = False,
    log_sink: Optional[Callable[[str], None]] = None,
) -> AgentRuntime:
    load_environment()

    uri = _require_env("NEO4J_URI")
    user = _require_env("NEO4J_USERNAME")
    password = _require_env("NEO4J_PASSWORD")
    azure_endpoint = _require_env("AZURE_ENDPOINT")
    azure_api_key = _require_env("AZURE_API_KEY")
    azure_api_version = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
    azure_deployment = os.environ.get("AZURE_MODEL_DEPLOYMENT", "gpt-5.2")
    llm_model = os.environ.get("AZURE_MODEL_NAME", "gpt-5.2")

    client = AzureOpenAIClient(
        endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        deployment=azure_deployment,
    )
    embed_fn = build_embed_fn()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    orchestrator = KGMultiAgentOrchestrator(
        driver,
        client,
        embed_fn,
        model=llm_model,
        log_progress=log_progress,
        log_sink=log_sink,
        use_kg=use_kg,
    )
    return AgentRuntime(driver=driver, orchestrator=orchestrator, embed_fn=embed_fn)
