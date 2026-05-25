from __future__ import annotations

import hashlib
import io
from typing import Any, Callable, Dict, Iterable, List, Sequence


def _chunk_text(text: str, *, chunk_size: int = 1600, overlap: int = 250) -> List[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def extract_pdf_records(
    file_bytes: bytes,
    *,
    filename: str,
    max_pages: int = 80,
    max_chunks_per_page: int = 4,
) -> List[Dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency `pypdf`. Install it with `pip install pypdf`.") from exc

    reader = PdfReader(io.BytesIO(file_bytes))
    rows: List[Dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages[:max_pages], start=1):
        extracted_text = (page.extract_text() or "").strip()
        if not extracted_text:
            continue
        page_chunks = _chunk_text(extracted_text)[:max_chunks_per_page]
        for chunk_index, chunk in enumerate(page_chunks, start=1):
            digest = hashlib.sha1(f"{filename}:{page_index}:{chunk_index}:{chunk[:160]}".encode("utf-8")).hexdigest()
            rows.append(
                {
                    "source_type": "uploaded_pdf",
                    "source_name": filename,
                    "page": str(page_index),
                    "chunk_index": str(chunk_index),
                    "chunk_uuid": digest,
                    "text": chunk,
                }
            )
    return rows


def rank_records_by_question(
    records: Sequence[Dict[str, Any]],
    *,
    question: str,
    embed_fn: Callable[[List[str]], List[List[float]]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    usable = [record for record in records if isinstance(record.get("text"), str) and record.get("text", "").strip()]
    if not usable:
        return []

    texts = [str(record["text"]) for record in usable]
    query_embedding = embed_fn([question])[0]
    text_embeddings = embed_fn(texts)

    scored: List[tuple[float, Dict[str, Any]]] = []
    for record, embedding in zip(usable, text_embeddings):
        score = _dot(query_embedding, embedding)
        enriched = dict(record)
        enriched["similarity"] = f"{score:.4f}"
        scored.append((score, enriched))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _score, record in scored[:top_k]]


def summarize_sources(records: Iterable[Dict[str, Any]]) -> str:
    names = []
    for record in records:
        source_name = record.get("source_name")
        if isinstance(source_name, str) and source_name.strip():
            names.append(source_name.strip())
    deduped = list(dict.fromkeys(names))
    return ", ".join(deduped)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))
