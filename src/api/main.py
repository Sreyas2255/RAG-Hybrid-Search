from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.bm25_retriever import create_bm25_index
from src.generation.rag_pipeline import rag_answer, get_reranker


# ============================================================
# GLOBAL RAG RESOURCES
# ============================================================

bm25 = None
chunks = None


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global bm25
    global chunks

    print("\n========================================")
    print("        STARTING RAG API")
    print("========================================")

    try:
        print("\nCreating BM25 index...")

        bm25, chunks = create_bm25_index()

        print("\nBM25 index ready.")
        print(f"Chunks loaded: {len(chunks)}")

        print("\nLoading reranker model...")

        get_reranker()

        print("\nReranker ready.")

        print("\n========================================")
        print("        RAG API READY")
        print("========================================")

    except Exception as error:

        print(
            f"\nRAG startup error: "
            f"{type(error).__name__}: {error}"
        )

        raise

    yield

    print("\n========================================")
    print("        SHUTTING DOWN RAG API")
    print("========================================")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Production-style Retrieval-Augmented Generation API "
        "using dense retrieval, BM25, RRF, cross-encoder reranking, "
        "and grounded generation with citations."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the indexed documents.",
        examples=[
            "What is a machine learning algorithm?"
        ],
    )


# ============================================================
# LEGACY RESPONSE MODEL
# Keeps the existing /ask API compatible
# ============================================================

class QuestionResponse(BaseModel):
    question: str
    answer: str


# ============================================================
# STRUCTURED RESPONSE MODELS
# Used by the new /v1/ask endpoint
# ============================================================

class SourceMetadata(BaseModel):
    source: Optional[str] = None
    page: Optional[Any] = None
    section: Optional[str] = None
    chunk_id: Optional[Any] = None


class Citation(BaseModel):
    citation_id: int
    source: Optional[str] = None
    page: Optional[Any] = None
    section: Optional[str] = None
    chunk_id: Optional[Any] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Optional[float] = None
    sources: list[SourceMetadata] = Field(default_factory=list)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely read a value from either:

    - dictionary
    - Pydantic model
    - normal Python object
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _normalize_pipeline_result(
    result: Any,
) -> dict[str, Any]:
    """
    Normalize different possible rag_pipeline return formats.

    Supported:

    1. String:
        "The answer is ... [1]"

    2. Dictionary:
        {
            "answer": "...",
            "citations": [...],
            "confidence": 0.91,
            "sources": [...]
        }

    3. Object:
        object.answer
        object.citations
        object.confidence
        object.sources
    """

    # --------------------------------------------------------
    # STRING RESULT
    # --------------------------------------------------------

    if isinstance(result, str):

        return {
            "answer": result,
            "citations": [],
            "confidence": None,
            "sources": [],
        }

    # --------------------------------------------------------
    # DICTIONARY / OBJECT RESULT
    # --------------------------------------------------------

    answer = _get_value(
        result,
        "answer",
        "",
    )

    citations = _get_value(
        result,
        "citations",
        [],
    )

    confidence = _get_value(
        result,
        "confidence",
        None,
    )

    sources = _get_value(
        result,
        "sources",
        [],
    )

    # Some pipelines may call the field "source_metadata".
    if not sources:

        sources = _get_value(
            result,
            "source_metadata",
            [],
        )

    # Some pipelines may use "confidence_score".
    if confidence is None:

        confidence = _get_value(
            result,
            "confidence_score",
            None,
        )

    return {
        "answer": str(answer),
        "citations": citations or [],
        "confidence": confidence,
        "sources": sources or [],
    }


def _normalize_citations(
    citations: list[Any],
) -> list[Citation]:
    """
    Convert pipeline citation information into
    the API Citation schema.
    """

    normalized = []

    for index, citation in enumerate(citations, start=1):

        # --------------------------------------------
        # Citation as integer
        # --------------------------------------------

        if isinstance(citation, int):

            normalized.append(
                Citation(
                    citation_id=citation,
                )
            )

            continue

        # --------------------------------------------
        # Citation as string
        # --------------------------------------------

        if isinstance(citation, str):

            normalized.append(
                Citation(
                    citation_id=index,
                    source=citation,
                )
            )

            continue

        # --------------------------------------------
        # Citation as dictionary/object
        # --------------------------------------------

        citation_id = _get_value(
            citation,
            "citation_id",
            index,
        )

        source = _get_value(
            citation,
            "source",
        )

        page = _get_value(
            citation,
            "page",
        )

        section = _get_value(
            citation,
            "section",
        )

        chunk_id = _get_value(
            citation,
            "chunk_id",
        )

        normalized.append(
            Citation(
                citation_id=citation_id,
                source=source,
                page=page,
                section=section,
                chunk_id=chunk_id,
            )
        )

    return normalized


def _normalize_sources(
    sources: list[Any],
) -> list[SourceMetadata]:
    """
    Convert source information into a consistent API format.
    """

    normalized = []

    for source in sources:

        # --------------------------------------------
        # Source is a string
        # --------------------------------------------

        if isinstance(source, str):

            normalized.append(
                SourceMetadata(
                    source=source,
                )
            )

            continue

        # --------------------------------------------
        # Source is dictionary/object
        # --------------------------------------------

        normalized.append(
            SourceMetadata(
                source=_get_value(
                    source,
                    "source",
                    _get_value(
                        source,
                        "source_document",
                    ),
                ),
                page=_get_value(
                    source,
                    "page",
                ),
                section=_get_value(
                    source,
                    "section",
                ),
                chunk_id=_get_value(
                    source,
                    "chunk_id",
                ),
            )
        )

    return normalized


def _extract_chunk_metadata(
    chunk: Any,
) -> SourceMetadata:
    """
    Extract metadata from a stored chunk.

    This supports common chunk representations:

    - dictionary
    - object with metadata
    - LangChain Document
    """

    metadata = _get_value(
        chunk,
        "metadata",
        {},
    )

    if metadata is None:
        metadata = {}

    source = None
    page = None
    section = None
    chunk_id = None

    # --------------------------------------------
    # Direct chunk fields
    # --------------------------------------------

    source = _get_value(
        chunk,
        "source",
    )

    page = _get_value(
        chunk,
        "page",
    )

    section = _get_value(
        chunk,
        "section",
    )

    chunk_id = _get_value(
        chunk,
        "chunk_id",
    )

    # --------------------------------------------
    # Metadata fields
    # --------------------------------------------

    if isinstance(metadata, dict):

        source = source or metadata.get(
            "source"
        )

        source = source or metadata.get(
            "source_document"
        )

        page = (
            page
            if page is not None
            else metadata.get("page")
        )

        section = section or metadata.get(
            "section"
        )

        chunk_id = (
            chunk_id
            if chunk_id is not None
            else metadata.get("chunk_id")
        )

    return SourceMetadata(
        source=source,
        page=page,
        section=section,
        chunk_id=chunk_id,
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    API root endpoint.
    """

    return {
        "message": "Hybrid RAG API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    summary="Health check",
    description="Check whether the RAG resources are ready.",
)
def health():

    return {
        "status": "healthy",
        "bm25_ready": bm25 is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
    }


# ============================================================
# LEGACY ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question",
    description=(
        "Ask a question about the indexed documents. "
        "This endpoint is kept for backwards compatibility."
    ),
)
def ask_question(
    request: QuestionRequest,
):

    if bm25 is None or chunks is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        result = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
        )

        normalized = _normalize_pipeline_result(
            result
        )

        return QuestionResponse(
            question=question,
            answer=normalized["answer"],
        )

    except Exception as error:

        print(
            f"RAG error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while processing "
                "the question."
            ),
        )


# ============================================================
# VERSIONED ASK ENDPOINT
# ============================================================

@app.post(
    "/v1/ask",
    response_model=AskResponse,
    summary="Ask a question with citations and metadata",
    description=(
        "Run the hybrid RAG pipeline and return the grounded "
        "answer together with citations, confidence information, "
        "and source metadata."
    ),
)
def ask_question_v1(
    request: QuestionRequest,
):

    if bm25 is None or chunks is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        result = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
        )

        normalized = _normalize_pipeline_result(
            result
        )

        citations = _normalize_citations(
            normalized["citations"]
        )

        sources = _normalize_sources(
            normalized["sources"]
        )

        confidence = normalized["confidence"]

        # --------------------------------------------
        # Validate confidence if supplied
        # --------------------------------------------

        if confidence is not None:

            try:

                confidence = float(confidence)

                # Keep confidence inside 0-1.
                confidence = max(
                    0.0,
                    min(1.0, confidence),
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = None

        return AskResponse(
            question=question,
            answer=normalized["answer"],
            citations=citations,
            confidence=confidence,
            sources=sources,
        )

    except Exception as error:

        print(
            f"RAG v1 error: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while processing "
                "the question."
            ),
        )


# ============================================================
# DOCUMENTS ENDPOINT
# ============================================================

@app.get(
    "/v1/documents",
    summary="List indexed documents",
    description=(
        "Return source metadata for documents currently "
        "loaded into the RAG system."
    ),
)
def list_documents():

    if chunks is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    documents = {}

    for chunk in chunks:

        metadata = _extract_chunk_metadata(
            chunk
        )

        source = metadata.source

        if not source:

            continue

        if source not in documents:

            documents[source] = {
                "source": source,
                "pages": [],
                "sections": [],
                "chunk_count": 0,
            }

        documents[source]["chunk_count"] += 1

        if metadata.page is not None:

            if metadata.page not in documents[
                source
            ]["pages"]:

                documents[source]["pages"].append(
                    metadata.page
                )

        if metadata.section:

            if metadata.section not in documents[
                source
            ]["sections"]:

                documents[source][
                    "sections"
                ].append(
                    metadata.section
                )

    return {
        "count": len(documents),
        "documents": list(
            documents.values()
        ),
    }


# ============================================================
# INDEX STATUS
# ============================================================

@app.get(
    "/v1/status",
    summary="RAG system status",
)
def rag_status():

    return {
        "api_version": "1.1.0",
        "status": (
            "ready"
            if bm25 is not None and chunks is not None
            else "not_ready"
        ),
        "bm25_ready": bm25 is not None,
        "reranker_loaded": (
            True
            if bm25 is not None
            else False
        ),
        "chunks_loaded": (
            len(chunks)
            if chunks
            else 0
        ),
    }