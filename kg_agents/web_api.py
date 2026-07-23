from __future__ import annotations

import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pdf_context import extract_pdf_records, rank_records_by_question
from .runtime import AgentRuntime, create_runtime


class AskRequest(BaseModel):
    question: str = Field(default="", description="User question to send to the research agent.")
    use_kg: bool | None = Field(default=None, description="Override the runtime default for KG usage.")
    compare_baseline: bool = Field(default=False, description="Also run the non-KG baseline for side-by-side output.")
    interactive: bool = Field(
        default=False,
        description="If true, pause for elicitation clarification and human collaboration gates.",
    )
    session_id: str | None = Field(default=None, description="Resume a paused interactive session.")
    clarification_answers: str | None = Field(
        default=None,
        description="Answers to clarifying questions when resuming elicitation.",
    )
    human_feedback: str | None = Field(
        default=None,
        description="Human collaboration feedback: continue | stop | free text.",
    )


class AskResponse(BaseModel):
    answer: str
    iterations_used: int
    attached_files: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(
        default="complete",
        description="complete | needs_clarification | awaiting_human_feedback",
    )
    research_intent: dict[str, Any] | None = None
    knowledge_package: dict[str, Any] | None = None
    session_id: str | None = None
    clarifying_questions: list[str] | None = None
    collaboration_prompt: str | None = None


_runtime_lock = Lock()


def _allowed_origins() -> list[str]:
    configured = os.environ.get("WEB_ALLOWED_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.runtime = None
    app.state.startup_error = None
    try:
        yield
    finally:
        runtime: AgentRuntime | None = getattr(app.state, "runtime", None)
        if runtime is not None:
            runtime.close()


app = FastAPI(
    title="T-Rex ISR Agent API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_runtime() -> AgentRuntime:
    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        raise HTTPException(status_code=500, detail=f"Backend startup failed: {startup_error}")

    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        with _runtime_lock:
            runtime = getattr(app.state, "runtime", None)
            if runtime is None:
                try:
                    runtime = create_runtime()
                except Exception as exc:  # pragma: no cover - depends on local credentials/services
                    app.state.startup_error = str(exc)
                    raise HTTPException(status_code=500, detail=f"Backend startup failed: {exc}") from exc
                app.state.runtime = runtime
    return runtime


@app.get("/")
def root() -> dict[str, object]:
    startup_error = getattr(app.state, "startup_error", None)
    runtime_ready = getattr(app.state, "runtime", None) is not None
    return {
        "service": "T-Rex ISR Agent API",
        "ok": startup_error is None,
        "runtime_ready": runtime_ready,
        "startup_error": startup_error,
        "routes": ["/health", "/ask", "/ask/continue"],
    }


@app.get("/health")
def healthcheck() -> dict[str, object]:
    startup_error = getattr(app.state, "startup_error", None)
    return {
        "ok": startup_error is None,
        "runtime_ready": getattr(app.state, "runtime", None) is not None,
        "startup_error": startup_error,
    }


def _to_ask_response(result: Any, *, attached_files: list[str] | None = None) -> AskResponse:
    state = getattr(result, "state", None)
    research_intent = getattr(state, "research_intent", None) if state is not None else None
    knowledge_package = getattr(state, "knowledge_package", None) if state is not None else None
    status = getattr(result, "status", None) or "complete"
    clarifying = getattr(result, "clarifying_questions", None)
    if clarifying is None and isinstance(research_intent, dict):
        clarifying = list(research_intent.get("clarifying_questions") or []) or None
    return AskResponse(
        answer=result.answer,
        iterations_used=result.iterations_used,
        attached_files=attached_files or [],
        debug=_build_debug_info(result),
        status=status,
        research_intent=research_intent if isinstance(research_intent, dict) else None,
        knowledge_package=knowledge_package if isinstance(knowledge_package, dict) else None,
        session_id=getattr(result, "session_id", None),
        clarifying_questions=clarifying,
        collaboration_prompt=getattr(result, "collaboration_prompt", None),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: Request) -> AskResponse:
    payload, uploaded_files = await _parse_request(request)

    # Resume via /ask when session_id is provided.
    if payload.session_id:
        runtime = _get_runtime()
        try:
            result = runtime.continue_ask(
                session_id=payload.session_id,
                clarification_answers=payload.clarification_answers,
                human_feedback=payload.human_feedback or payload.question or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Session continue failed: {exc}") from exc
        return _to_ask_response(result)

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    runtime = _get_runtime()
    try:
        document_records = await _extract_document_records(question, uploaded_files, runtime)
        result = runtime.ask(
            question,
            use_kg=payload.use_kg,
            compare_baseline=payload.compare_baseline,
            document_records=document_records,
            interactive=payload.interactive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - depends on network/credentials/runtime services
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

    return _to_ask_response(
        result,
        attached_files=[upload.filename or "uploaded.pdf" for upload in uploaded_files],
    )


@app.post("/ask/continue", response_model=AskResponse)
async def continue_question(request: Request) -> AskResponse:
    payload, _uploaded = await _parse_request(request)
    if not payload.session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")
    runtime = _get_runtime()
    try:
        result = runtime.continue_ask(
            session_id=payload.session_id,
            clarification_answers=payload.clarification_answers,
            human_feedback=payload.human_feedback or payload.question or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Session continue failed: {exc}") from exc
    return _to_ask_response(result)


async def _parse_request(request: Request) -> tuple[AskRequest, list[UploadFile]]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = AskRequest.model_validate(await request.json())
        return payload, []

    form = await request.form()
    files = [value for value in form.getlist("files") if hasattr(value, "filename") and hasattr(value, "read")]
    payload = AskRequest(
        question=str(form.get("question", "")),
        use_kg=_coerce_optional_bool(form.get("use_kg")),
        compare_baseline=_coerce_optional_bool(form.get("compare_baseline")) or False,
        interactive=_coerce_optional_bool(form.get("interactive")) or False,
        session_id=(str(form.get("session_id")) if form.get("session_id") not in (None, "") else None),
        clarification_answers=(
            str(form.get("clarification_answers"))
            if form.get("clarification_answers") not in (None, "")
            else None
        ),
        human_feedback=(
            str(form.get("human_feedback")) if form.get("human_feedback") not in (None, "") else None
        ),
    )
    return payload, files


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


async def _extract_document_records(
    question: str,
    uploaded_files: list[UploadFile],
    runtime: AgentRuntime,
) -> list[dict[str, Any]]:
    if not uploaded_files:
        return []

    parsed_records: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        filename = (uploaded_file.filename or "uploaded.pdf").strip() or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF uploads are supported right now: {filename}")

        raw_bytes = await uploaded_file.read()
        if not raw_bytes:
            continue

        file_records = extract_pdf_records(raw_bytes, filename=filename)
        parsed_records.extend(file_records)

    if not parsed_records:
        raise HTTPException(status_code=400, detail="No readable text was found in the uploaded PDF files.")

    ranked_records = rank_records_by_question(
        parsed_records,
        question=question,
        embed_fn=runtime.embed_fn,
        top_k=10,
    )
    if not ranked_records:
        raise HTTPException(status_code=400, detail="The uploaded PDFs did not yield any useful text passages.")
    return ranked_records


def _build_debug_info(result: Any) -> dict[str, Any]:
    state = getattr(result, "state", None)
    neo4j_uri = os.environ.get("NEO4J_URI", "").strip()
    parsed = urlparse(neo4j_uri) if neo4j_uri else None
    last_cypher = getattr(state, "last_cypher", None) if state is not None else None
    last_records = getattr(state, "last_records", []) if state is not None else []
    last_error = getattr(state, "last_error", None) if state is not None else None
    extracted_records = getattr(state, "extracted_records", []) if state is not None else []
    scratchpad = getattr(state, "scratchpad", []) if state is not None else []

    research_intent = getattr(state, "research_intent", None) if state is not None else None
    reflection = getattr(state, "reflection", None) if state is not None else None
    integration_state = getattr(state, "integration_state", None) if state is not None else None
    analysis_step = getattr(state, "analysis_step", None) if state is not None else None
    cycle_index = getattr(state, "cycle_index", None) if state is not None else None
    cycle_packages = getattr(state, "cycle_knowledge_packages", None) if state is not None else None
    internalization = getattr(state, "internalization_result", None) if state is not None else None

    elicitation_summary = None
    if isinstance(research_intent, dict):
        elicitation_summary = {
            "discovery_type": research_intent.get("discovery_type"),
            "is_sufficiently_specified": research_intent.get("is_sufficiently_specified"),
            "target_concepts": (research_intent.get("target_concepts") or [])[:8],
            "clarifying_questions": (research_intent.get("clarifying_questions") or [])[:4],
            "refined_question": (research_intent.get("refined_question") or "")[:300],
        }
    reflection_summary = None
    if isinstance(reflection, dict):
        reflection_summary = {
            "sufficient": reflection.get("sufficient"),
            "continue_loop": reflection.get("continue_loop"),
            "recommended_analyses": (reflection.get("recommended_analyses") or [])[:6],
            "rationale": (reflection.get("rationale") or "")[:400],
        }
    integration_summary = None
    if isinstance(integration_state, dict):
        integration_summary = {
            "confidence": integration_state.get("confidence"),
            "emerging_concepts": (integration_state.get("emerging_concepts") or [])[:8],
            "propositions": (integration_state.get("propositions") or [])[:6],
            "theoretical_gaps": (integration_state.get("theoretical_gaps") or [])[:6],
        }

    return {
        "neo4j_uri": neo4j_uri,
        "neo4j_host": parsed.hostname if parsed else None,
        "neo4j_scheme": parsed.scheme if parsed else None,
        "kg_query_attempted": bool(last_cypher or last_records or last_error),
        "kg_record_count": len(last_records) if isinstance(last_records, list) else 0,
        "pdf_record_count": len(extracted_records) if isinstance(extracted_records, list) else 0,
        "last_cypher_preview": (last_cypher or "")[:1200],
        "last_error": last_error,
        "iterations_used": getattr(result, "iterations_used", None),
        "scratchpad_tail": scratchpad[-8:] if isinstance(scratchpad, list) else [],
        "analysis_step": analysis_step,
        "cycle_index": cycle_index,
        "elicitation": elicitation_summary,
        "reflection": reflection_summary,
        "integration": integration_summary,
        "has_knowledge_package": bool(getattr(state, "knowledge_package", None)) if state is not None else False,
        "has_vicarious": bool(getattr(state, "vicarious", None)) if state is not None else False,
        "cycle_packages_count": len(cycle_packages) if isinstance(cycle_packages, list) else 0,
        "internalization": internalization,
        "status": getattr(result, "status", None),
        "session_id": getattr(result, "session_id", None),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("kg_agents.web_api:app", host="127.0.0.1", port=8000, reload=True)
