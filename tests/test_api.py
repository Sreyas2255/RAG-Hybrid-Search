import pytest
from fastapi.testclient import TestClient

import src.api.main as main


# ============================================================
# MOCK RAG RESOURCES
# ============================================================

def fake_create_bm25_index():
    """
    Fake BM25 index for testing.

    We do not want tests to rebuild the real
    3095-chunk BM25 index.
    """

    return "fake_bm25", ["fake_chunk_1", "fake_chunk_2"]


def fake_get_reranker():
    """
    Fake reranker for testing.

    Prevents the CrossEncoder model from
    being loaded during API tests.
    """

    return "fake_reranker"


# ============================================================
# TEST FIXTURE
# ============================================================

@pytest.fixture
def client(monkeypatch):

    # --------------------------------------------------------
    # Replace expensive real startup operations
    # --------------------------------------------------------

    monkeypatch.setattr(
        main,
        "create_bm25_index",
        fake_create_bm25_index,
    )

    monkeypatch.setattr(
        main,
        "get_reranker",
        fake_get_reranker,
    )

    # --------------------------------------------------------
    # Create FastAPI test client
    # --------------------------------------------------------

    with TestClient(main.app) as test_client:

        yield test_client


# ============================================================
# ROOT ENDPOINT
# ============================================================

def test_root(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
    "message": "Hybrid RAG API is running",
    "version": "1.0.0",
    "docs": "/docs",
    "health": "/health",
}


# ============================================================
# HEALTH ENDPOINT
# ============================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["bm25_ready"] is True

    assert data["chunks_loaded"] == 2


# ============================================================
# ASK ENDPOINT
# ============================================================

def test_ask_question(client, monkeypatch):

    # --------------------------------------------------------
    # Fake RAG answer
    # --------------------------------------------------------

    def fake_rag_answer(
        question,
        bm25,
        chunks,
    ):

        assert question == "What is machine learning?"

        assert bm25 == "fake_bm25"

        assert chunks == [
            "fake_chunk_1",
            "fake_chunk_2",
        ]

        return "Machine learning is learning from data."

    monkeypatch.setattr(
        main,
        "rag_answer",
        fake_rag_answer,
    )

    # --------------------------------------------------------
    # Send API request
    # --------------------------------------------------------

    response = client.post(
        "/ask",
        json={
            "question": "What is machine learning?"
        },
    )

    # --------------------------------------------------------
    # Verify response
    # --------------------------------------------------------

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What is machine learning?"
    )

    assert data["answer"] == (
        "Machine learning is learning from data."
    )


# ============================================================
# EMPTY QUESTION
# ============================================================

def test_empty_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "   "
        },
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Question cannot be empty."
    )


# ============================================================
# MISSING QUESTION
# ============================================================

def test_missing_question(client):

    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422


# ============================================================
# INVALID QUESTION TYPE
# ============================================================

def test_invalid_question_type(client):

    response = client.post(
        "/ask",
        json={
            "question": 12345
        },
    )

    assert response.status_code == 422


# ============================================================
# RAG ERROR
# ============================================================

def test_rag_error(client, monkeypatch):

    # --------------------------------------------------------
    # Simulate RAG failure
    # --------------------------------------------------------

    def fake_rag_answer(
        question,
        bm25,
        chunks,
    ):

        raise RuntimeError(
            "Test RAG failure"
        )

    monkeypatch.setattr(
        main,
        "rag_answer",
        fake_rag_answer,
    )

    # --------------------------------------------------------
    # Send request
    # --------------------------------------------------------

    response = client.post(
        "/ask",
        json={
            "question": "What is machine learning?"
        },
    )

    # --------------------------------------------------------
    # Verify HTTP 500
    # --------------------------------------------------------

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "An error occurred while processing the question."
    )