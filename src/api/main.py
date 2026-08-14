from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.bm25_retriever import (
    create_bm25_index,
)

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
    # Create BM25 index
    # --------------------------------------------------------

    print("\nCreating BM25 index...")

    bm25, chunks = create_bm25_index()

    print("\nBM25 index ready.")
    print(f"Chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # Load Cross-Encoder reranker
    # --------------------------------------------------------

    print("\nLoading reranker model...")

    get_reranker()

    print("\nReranker ready.")

    print("\n========================================")
    print("        RAG API READY")
    print("========================================")

    # --------------------------------------------------------
    # Application runs here
    # --------------------------------------------------------

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    print("\n========================================")
    print("        SHUTTING DOWN RAG API")
    print("========================================")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Document Question Answering API "
        "using Dense Retrieval, BM25, "
        "Hybrid RRF, Cross-Encoder Reranking, "
        "and Groq."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    """
    Request body for the /ask endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        description=(
            "Question to ask the document "
            "question-answering system."
        ),
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class QuestionResponse(BaseModel):
    """
    Response returned by the /ask endpoint.
    """

    question: str
    answer: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Basic API information endpoint.
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

@app.get("/health")
def health():
    """
    Check whether the RAG resources are ready.
    """

    return {
        "status": "healthy",
        "bm25_ready": bm25 is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest,
):
    """
    Ask a question about the indexed documents.

    Pipeline:

        Question
            ↓
        Dense Retrieval
            ↓
        BM25
            ↓
        RRF Hybrid Retrieval
            ↓
        Cross-Encoder Reranking
            ↓
        Context
            ↓
        Groq
            ↓
        Answer
    """

    # ========================================================
    # CHECK RAG READINESS
    # ========================================================

    if bm25 is None or chunks is None:

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    # ========================================================
    # CLEAN QUESTION
    # ========================================================

    question = request.question.strip()

    # ========================================================
    # EMPTY QUESTION CHECK
    # ========================================================

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # ========================================================
    # RUN RAG PIPELINE
    # ========================================================

    try:

        answer = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
        )

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return QuestionResponse(
            question=question,
            answer=answer,
        )

    except Exception as error:

        # ----------------------------------------------------
        # Log actual error on server
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "RAG ERROR"
        )

        print(
            "========================================"
        )

        print(
            f"Error type: {type(error).__name__}"
        )

        print(
            f"Error: {error}"
        )

        # ----------------------------------------------------
        # Do not expose internal error details to API client
        # ----------------------------------------------------

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the question."
            ),
        )