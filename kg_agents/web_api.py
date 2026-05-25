from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pdf_context import extract_pdf_records, rank_records_by_question
from .runtime import AgentRuntime, create_runtime


class AskRequest(BaseModel):
    question: str = Field(description="User question to send to the research agent.")
    use_kg: bool | None = Field(default=None, description="Override the runtime default for KG usage.")
    compare_baseline: bool = Field(default=False, description="Also run the non-KG baseline for side-by-side output.")


class AskResponse(BaseModel):
    answer: str
    iterations_used: int
    attached_files: list[str] = Field(default_factory=list)


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
    runtime: AgentRuntime | None = None
    app.state.runtime = None
    app.state.startup_error = None
    try:
        runtime = create_runtime()
        app.state.runtime = runtime
    except Exception as exc:  # pragma: no cover - runtime depends on local credentials/services
        app.state.startup_error = str(exc)
    try:
        yield
    finally:
        if runtime is not None:
            runtime.close()


app = FastAPI(
    title="T-Rex ISR Agent API",
    version="0.1.0",
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
        raise HTTPException(status_code=503, detail="Backend runtime is not ready yet.")
    return runtime


@app.get("/health")
def healthcheck() -> dict[str, object]:
    startup_error = getattr(app.state, "startup_error", None)
    return {
        "ok": startup_error is None,
        "startup_error": startup_error,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: Request) -> AskResponse:
    payload, uploaded_files = await _parse_request(request)
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
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - depends on network/credentials/runtime services
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        iterations_used=result.iterations_used,
        attached_files=[upload.filename or "uploaded.pdf" for upload in uploaded_files],
    )


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("kg_agents.web_api:app", host="127.0.0.1", port=8000, reload=True)
