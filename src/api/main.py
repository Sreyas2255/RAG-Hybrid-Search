from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.bm25_retriever import create_bm25_index

from src.generation.rag_pipeline import (
    rag_answer,
    get_reranker,
)


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

    # --------------------------------------------------------
    # BUILD BM25 INDEX
    # --------------------------------------------------------

    print("\nCreating BM25 index...")

    bm25, chunks = create_bm25_index()

    print("\nBM25 index ready.")
    print(f"Chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # LOAD RERANKER
    # --------------------------------------------------------

    print("\nLoading reranker model...")

    get_reranker()

    print("\nReranker ready.")

    print("\n========================================")
    print("        RAG API READY")
    print("========================================")

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    print("\n========================================")
    print("        SHUTTING DOWN RAG API")
    print("========================================")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hybrid RAG API",
    description="Document Question Answering API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system.",
    )


# ============================================================
# LEGACY RESPONSE MODEL
# ============================================================

class QuestionResponse(BaseModel):

    question: str
    answer: str


# ============================================================
# V1 RESPONSE MODEL
# ============================================================

class V1QuestionResponse(BaseModel):

    question: str

    answer: str

    citations: list[Any] = Field(
        default_factory=list
    )

    confidence: float | None = None

    grounded: bool = True

    answer_source: str = "documents"

    sources: list[Any] = Field(
        default_factory=list
    )

    citation_verification: dict[str, Any] | None = None

    # IMPORTANT:
    #
    # The RAG pipeline currently returns this as an INTEGER.
    #
    # Example:
    #
    #     retrieved_chunks = 5
    #
    # This means 5 chunks were retrieved/reranked.
    #
    # It is NOT a list of chunk objects.
    #
    retrieved_chunks: int = 0


# ============================================================
# HELPERS
# ============================================================

def normalize_pipeline_result(
    result,
    question: str,
):
    """
    Normalize the result returned by rag_answer().

    The structured RAG pipeline should return a dictionary
    when return_details=True.

    The current pipeline returns:

        question
        answer
        citations
        confidence
        sources
        citation_verification
        retrieved_chunks

    IMPORTANT:
        retrieved_chunks is currently an integer count,
        e.g. 5.

    This function also safely handles a plain string result.
    """

    # --------------------------------------------------------
    # Structured result
    # --------------------------------------------------------

    if isinstance(result, dict):

        # ----------------------------------------------------
        # Normalize retrieved_chunks
        # ----------------------------------------------------

        retrieved_chunks = result.get(
            "retrieved_chunks",
            0,
        )

        # The current pipeline returns an integer.
        #
        # We also protect the API in case the pipeline
        # returns None or another unexpected value.
        #

        if retrieved_chunks is None:

            retrieved_chunks = 0

        elif isinstance(
            retrieved_chunks,
            int,
        ):

            pass

        elif isinstance(
            retrieved_chunks,
            list,
        ):

            # Backward-compatible support if the pipeline
            # is changed later to return the actual chunks.
            retrieved_chunks = len(
                retrieved_chunks
            )

        else:

            try:

                retrieved_chunks = int(
                    retrieved_chunks
                )

            except (
                TypeError,
                ValueError,
            ):

                retrieved_chunks = 0

        # ----------------------------------------------------
        # Normalize citations
        # ----------------------------------------------------

        citations = result.get(
            "citations",
            [],
        )

        if citations is None:

            citations = []

        elif not isinstance(
            citations,
            list,
        ):

            citations = [citations]

        # ----------------------------------------------------
        # Normalize sources
        # ----------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        if sources is None:

            sources = []

        elif not isinstance(
            sources,
            list,
        ):

            sources = [sources]

        # ----------------------------------------------------
        # Return normalized result
        # ----------------------------------------------------

        return {
            "question": result.get(
                "question",
                question,
            ),

            "answer": str(
                result.get(
                    "answer",
                    "",
                )
            ),

            "citations": citations,

            "confidence": result.get(
                "confidence",
            ),

            "grounded": bool(
                result.get(
                    "grounded",
                    True,
                )
            ),

            "answer_source": str(
                result.get(
                    "answer_source",
                    "documents",
                )
            ),

            "sources": sources,

            "citation_verification": result.get(
                "citation_verification",
            ),

            "retrieved_chunks": retrieved_chunks,
        }

    # --------------------------------------------------------
    # Plain string fallback
    # --------------------------------------------------------

    return {
        "question": question,

        "answer": str(result),

        "citations": [],

        "confidence": None,

        "grounded": True,

        "answer_source": "documents",

        "sources": [],

        "citation_verification": None,

        "retrieved_chunks": 0,
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Hybrid RAG API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "bm25_ready": (
            bm25 is not None
        ),

        "chunks_loaded": (
            len(chunks)
            if chunks
            else 0
        ),
    }


# ============================================================
# STATUS ENDPOINT
# ============================================================

@app.get("/v1/status")
def status():

    # --------------------------------------------------------
    # Check reranker
    # --------------------------------------------------------

    try:

        import src.generation.rag_pipeline as rag_pipeline

        reranker_loaded = (
            getattr(
                rag_pipeline,
                "_reranker",
                None,
            )
            is not None
        )

    except Exception:

        reranker_loaded = False

    # --------------------------------------------------------
    # Return status
    # --------------------------------------------------------

    return {

        "status": (
            "ready"
            if (
                bm25 is not None
                and chunks is not None
            )
            else "not_ready"
        ),

        "bm25_ready": (
            bm25 is not None
        ),

        "chunks_loaded": (
            len(chunks)
            if chunks
            else 0
        ),

        "reranker_loaded": reranker_loaded,
    }


# ============================================================
# DOCUMENTS ENDPOINT
# ============================================================

@app.get("/v1/documents")
def documents():

    if chunks is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    documents_map = {}

    # --------------------------------------------------------
    # Extract document information from chunks
    # --------------------------------------------------------

    for chunk in chunks:

        metadata = {}

        # ----------------------------------------------------
        # Dictionary chunk
        # ----------------------------------------------------

        if isinstance(
            chunk,
            dict,
        ):

            metadata = chunk.get(
                "metadata",
                {},
            ) or {}

        # ----------------------------------------------------
        # Object chunk
        # ----------------------------------------------------

        elif hasattr(
            chunk,
            "metadata",
        ):

            metadata = (
                chunk.metadata
                or {}
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
        )

        section = metadata.get(
            "section",
        )

        # ----------------------------------------------------
        # Create document entry
        # ----------------------------------------------------

        if source not in documents_map:

            documents_map[source] = {

                "source": source,

                "chunk_count": 0,

                "pages": set(),

                "sections": set(),
            }

        documents_map[source][
            "chunk_count"
        ] += 1

        # ----------------------------------------------------
        # Page
        # ----------------------------------------------------

        if page is not None:

            documents_map[source][
                "pages"
            ].add(
                page
            )

        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        if section:

            documents_map[source][
                "sections"
            ].add(
                section
            )

    # --------------------------------------------------------
    # Convert sets to JSON-compatible lists
    # --------------------------------------------------------

    documents_list = []

    for document in documents_map.values():

        document["pages"] = sorted(
            document["pages"],
            key=lambda value: str(value),
        )

        document["sections"] = sorted(
            document["sections"],
            key=lambda value: str(value),
        )

        documents_list.append(
            document
        )

    # --------------------------------------------------------
    # Sort documents
    # --------------------------------------------------------

    documents_list.sort(
        key=lambda item: item["source"]
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "count": len(
            documents_list
        ),

        "documents": documents_list,
    }


# ============================================================
# LEGACY ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):

    if (
        bm25 is None
        or chunks is None
    ):

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

        # ----------------------------------------------------
        # Legacy endpoint
        # ----------------------------------------------------

        answer = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
        )

        # ----------------------------------------------------
        # Return only answer
        # ----------------------------------------------------

        return QuestionResponse(

            question=question,

            answer=(
                answer
                if isinstance(
                    answer,
                    str,
                )
                else str(answer)
            ),
        )

    except Exception as error:

        print(
            "\n========================================"
        )

        print(
            "        LEGACY RAG ERROR"
        )

        print(
            "========================================"
        )

        print(
            f"Type: {type(error).__name__}"
        )

        print(
            f"Error: {error}"
        )

        print(
            "========================================"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the question."
            ),
        )


# ============================================================
# V1 ASK ENDPOINT
# ============================================================

@app.post(
    "/v1/ask",
    response_model=V1QuestionResponse,
)
def ask_question_v1(
    request: QuestionRequest,
):

    # --------------------------------------------------------
    # Check RAG readiness
    # --------------------------------------------------------

    if (
        bm25 is None
        or chunks is None
    ):

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    # --------------------------------------------------------
    # Clean question
    # --------------------------------------------------------

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        print(
            "\n========================================"
        )

        print(
            "        V1 RAG REQUEST"
        )

        print(
            "========================================"
        )

        print(
            f"Question: {question}"
        )

        # ====================================================
        # RUN RAG PIPELINE
        # ====================================================

        result = rag_answer(

            question=question,

            bm25=bm25,

            chunks=chunks,

            return_details=True,
        )

        # ====================================================
        # NORMALIZE RESULT
        # ====================================================

        normalized = normalize_pipeline_result(

            result=result,

            question=question,
        )

        # ====================================================
        # LOG RESULT
        # ====================================================

        print(
            "\nV1 RAG RESULT:"
        )

        print(
            f"Answer length: "
            f"{len(normalized['answer'])}"
        )

        print(
            f"Citations: "
            f"{len(normalized['citations'])}"
        )

        print(
            f"Sources: "
            f"{len(normalized['sources'])}"
        )

        print(
            f"Retrieved chunks: "
            f"{normalized['retrieved_chunks']}"
        )

        print(
            f"Confidence: "
            f"{normalized['confidence']}"
        )

        print(
            f"Grounded: "
            f"{normalized['grounded']}"
        )

        print(
            f"Answer source: "
            f"{normalized['answer_source']}"
        )

        print(
            "========================================"
        )

        # ====================================================
        # RETURN STRUCTURED RESPONSE
        # ====================================================

        return V1QuestionResponse(

            question=normalized[
                "question"
            ],

            answer=normalized[
                "answer"
            ],

            citations=normalized[
                "citations"
            ],

            confidence=normalized[
                "confidence"
            ],

            grounded=normalized[
                "grounded"
            ],

            answer_source=normalized[
                "answer_source"
            ],

            sources=normalized[
                "sources"
            ],

            citation_verification=normalized[
                "citation_verification"
            ],

            retrieved_chunks=normalized[
                "retrieved_chunks"
            ],
        )

    # --------------------------------------------------------
    # HTTP exception
    # --------------------------------------------------------

    except HTTPException:

        raise

    # --------------------------------------------------------
    # Unexpected exception
    # --------------------------------------------------------

    except Exception as error:

        print(
            "\n========================================"
        )

        print(
            "        V1 RAG ERROR"
        )

        print(
            "========================================"
        )

        print(
            f"Type: {type(error).__name__}"
        )

        print(
            f"Error: {error}"
        )

        print(
            "========================================"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "An error occurred while "
                "processing the question."
            ),
        )