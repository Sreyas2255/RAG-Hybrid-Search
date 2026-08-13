
import os

from sentence_transformers import CrossEncoder

from src.retrieval.dense_retriever import (
    dense_search,
)

from src.retrieval.bm25_retriever import (
    create_bm25_index,
    bm25_search,
)

from src.generation.groq_llm import (
    generate_answer,
)


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_TOP_K = 50
BM25_TOP_K = 50

HYBRID_TOP_K = 30

RERANK_TOP_K = 5

RRF_K = 60

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# LOAD RERANKER
# ============================================================

_reranker = None


def get_reranker():
    """
    Load the cross-encoder reranker only once.
    """

    global _reranker

    if _reranker is None:

        print(
            "\nLoading reranker model..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print(
            "Reranker model loaded successfully."
        )

    return _reranker


# ============================================================
# RESULT TEXT EXTRACTION
# ============================================================

def get_result_text(result):
    """
    Extract text from a retrieval result.

    Supports the result formats used by the
    dense and BM25 retrievers.
    """

    # --------------------------------------------------------
    # result["text"]
    # --------------------------------------------------------

    if result.get("text"):

        return str(
            result["text"]
        )

    # --------------------------------------------------------
    # result["document"]
    # --------------------------------------------------------

    document = result.get(
        "document"
    )

    if document is not None:

        if hasattr(
            document,
            "page_content",
        ):

            return document.page_content

        if isinstance(
            document,
            str,
        ):

            return document

    # --------------------------------------------------------
    # result["page_content"]
    # --------------------------------------------------------

    if result.get(
        "page_content"
    ):

        return str(
            result["page_content"]
        )

    return ""


# ============================================================
# RESULT KEY
# ============================================================

def get_result_key(result):
    """
    Create a stable identifier for the same
    document chunk across dense and BM25 results.

    We prefer the chunk ID.

    If the ID is unavailable, we fall back to
    source + page + text.
    """

    # --------------------------------------------------------
    # Try ID
    # --------------------------------------------------------

    result_id = result.get(
        "id"
    )

    if result_id:

        return str(
            result_id
        )

    # --------------------------------------------------------
    # Metadata fallback
    # --------------------------------------------------------

    metadata = result.get(
        "metadata",
        {},
    )

    source = metadata.get(
        "source",
        "",
    )

    page = metadata.get(
        "page",
        "",
    )

    text = get_result_text(
        result
    )

    return (
        f"{source}|"
        f"{page}|"
        f"{text[:200]}"
    )


# ============================================================
# RRF HYBRID RETRIEVAL
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=HYBRID_TOP_K,
    k=RRF_K,
):
    """
    Combine dense and BM25 rankings using
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    Documents appearing in both retrieval
    systems receive contributions from both.
    """

    fused = {}

    # ========================================================
    # DENSE RESULTS
    # ========================================================

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        key = get_result_key(
            result
        )

        if key not in fused:

            fused[key] = {
                "id": result.get(
                    "id"
                ),
                "text": get_result_text(
                    result
                ),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_score": result.get(
                    "score"
                ),
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # ========================================================
    # BM25 RESULTS
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        key = get_result_key(
            result
        )

        if key not in fused:

            fused[key] = {
                "id": result.get(
                    "id"
                ),
                "text": get_result_text(
                    result
                ),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_score": None,
                "bm25_score": result.get(
                    "score"
                ),
                "rrf_score": 0.0,
            }

        else:

            fused[key]["bm25_rank"] = rank

            fused[key]["bm25_score"] = (
                result.get("score")
            )

            # ------------------------------------------------
            # Update text/document if missing
            # ------------------------------------------------

            if not fused[key].get(
                "text"
            ):

                fused[key]["text"] = (
                    get_result_text(
                        result
                    )
                )

            if not fused[key].get(
                "document"
            ):

                fused[key]["document"] = (
                    result.get(
                        "document"
                    )
                )

            if not fused[key].get(
                "metadata"
            ):

                fused[key]["metadata"] = (
                    result.get(
                        "metadata",
                        {},
                    )
                )

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # ========================================================
    # SORT BY RRF SCORE
    # ========================================================

    results = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return results[
        :top_k
    ]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def create_hybrid_results(
    query,
    bm25,
    chunks,
    top_k=HYBRID_TOP_K,
):
    """
    Run:

        Dense
        +
        BM25
        ↓
        RRF
    """

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "RUNNING DENSE RETRIEVAL"
    )

    print(
        "=============================="
    )

    dense_results = dense_search(
        query,
        top_k=DENSE_TOP_K,
    )

    print(
        f"Dense results: "
        f"{len(dense_results)}"
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "RUNNING BM25 RETRIEVAL"
    )

    print(
        "=============================="
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=BM25_TOP_K,
    )

    print(
        f"BM25 results: "
        f"{len(bm25_results)}"
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "COMBINING WITH RRF"
    )

    print(
        "=============================="
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k,
        k=RRF_K,
    )

    print(
        f"Hybrid results: "
        f"{len(hybrid_results)}"
    )

    return hybrid_results


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    query,
    results,
    top_k=RERANK_TOP_K,
):
    """
    Rerank hybrid candidates using
    CrossEncoder/ms-marco-MiniLM-L-6-v2.
    """

    if not results:

        return []

    model = get_reranker()

    print(
        f"\nReranking "
        f"{len(results)} candidates..."
    )

    # --------------------------------------------------------
    # Prepare query-document pairs
    # --------------------------------------------------------

    pairs = []

    valid_results = []

    for result in results:

        text = get_result_text(
            result
        )

        if not text.strip():
            continue

        pairs.append(
            [
                query,
                text,
            ]
        )

        valid_results.append(
            result
        )

    if not pairs:

        return []

    # --------------------------------------------------------
    # Cross-encoder scores
    # --------------------------------------------------------

    scores = model.predict(
        pairs
    )

    # --------------------------------------------------------
    # Attach reranker score
    # --------------------------------------------------------

    reranked = []

    for result, score in zip(
        valid_results,
        scores,
    ):

        item = dict(
            result
        )

        item[
            "reranker_score"
        ] = float(score)

        reranked.append(
            item
        )

    # --------------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------------

    reranked.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked[
        :top_k
    ]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):
    """
    Convert reranked documents into
    context for Groq.
    """

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        human_page = metadata.get(
            "human_page"
        )

        text = get_result_text(
            result
        )

        if not text.strip():

            continue

        if human_page is not None:

            page_info = (
                f"Page: {page} "
                f"(Human page: {human_page})"
            )

        else:

            page_info = (
                f"Page: {page}"
            )

        reranker_score = result.get(
            "reranker_score"
        )

        if reranker_score is not None:

            score_info = (
                f"\nReranker score: "
                f"{reranker_score:.4f}"
            )

        else:

            score_info = ""

        context_parts.append(
            f"""
--- Context {i} ---

Source: {source}

{page_info}{score_info}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# RAG PROMPT
# ============================================================

def create_rag_prompt(
    question,
    context,
):
    """
    Create a grounded RAG prompt.
    """

    return f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:

1. Use only information contained in the
   supplied context.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Do not make assumptions that are not
   supported by the documents.

5. Prefer the context that directly answers
   the question.

6. Give a clear and concise answer.

7. If several context sections support the
   answer, combine them carefully.

8. When useful, mention the source document
   and page number.

9. If the answer is not present in the
   provided context, say exactly:

"I could not find the answer in the provided documents."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# ============================================================
# COMPLETE RAG ANSWER
# ============================================================

def rag_answer(
    question,
    bm25,
    chunks,
):
    """
    Complete RAG pipeline:

        Question
            ↓
        Dense Retrieval
            ↓
        BM25
            ↓
        RRF
            ↓
        Cross-Encoder
            ↓
        Top 5
            ↓
        Context
            ↓
        Groq
            ↓
        Answer
    """

    # ========================================================
    # STEP 1: HYBRID RETRIEVAL
    # ========================================================

    hybrid_results = create_hybrid_results(
        query=question,
        bm25=bm25,
        chunks=chunks,
        top_k=HYBRID_TOP_K,
    )

    if not hybrid_results:

        return (
            "I could not find the answer "
            "in the provided documents."
        )

    # ========================================================
    # STEP 2: RERANK
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "RUNNING CROSS-ENCODER RERANKER"
    )

    print(
        "=============================="
    )

    reranked_results = rerank_results(
        question,
        hybrid_results,
        top_k=RERANK_TOP_K,
    )

    print(
        f"Reranked results: "
        f"{len(reranked_results)}"
    )

    if not reranked_results:

        return (
            "I could not find the answer "
            "in the provided documents."
        )

    # ========================================================
    # STEP 3: DISPLAY RERANKED RESULTS
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "RERANKED RESULTS"
    )

    print(
        "=============================="
    )

    for i, result in enumerate(
        reranked_results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print(
            f"\n--- Result {i} ---"
        )

        print(
            "ID:",
            result.get(
                "id"
            ),
        )

        print(
            "Reranker Score:",
            result.get(
                "reranker_score"
            ),
        )

        print(
            "RRF Score:",
            result.get(
                "rrf_score"
            ),
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "Unknown",
            ),
        )

    # ========================================================
    # STEP 4: BUILD CONTEXT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "BUILDING CONTEXT"
    )

    print(
        "=============================="
    )

    context = build_context(
        reranked_results
    )

    if not context.strip():

        return (
            "I could not find the answer "
            "in the provided documents."
        )

    # ========================================================
    # STEP 5: SHOW CONTEXT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "CONTEXT SENT TO GROQ"
    )

    print(
        "=============================="
    )

    print(
        context
    )

    # ========================================================
    # STEP 6: CREATE PROMPT
    # ========================================================

    prompt = create_rag_prompt(
        question=question,
        context=context,
    )

    # ========================================================
    # STEP 7: GROQ
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "SENDING CONTEXT TO GROQ"
    )

    print(
        "=============================="
    )

    answer = generate_answer(
        prompt
    )

    return answer


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "             RAG PIPELINE"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Build BM25 index
    # --------------------------------------------------------

    print(
        "\nCreating BM25 index..."
    )

    bm25, chunks = create_bm25_index()

    print(
        "\nBM25 index ready."
    )

    # --------------------------------------------------------
    # Question loop
    # --------------------------------------------------------

    while True:

        question = input(
            "\nEnter your question "
            "(or 'exit' to quit): "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nExiting RAG pipeline."
            )

            break

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:

            continue

        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        try:

            answer = rag_answer(
                question=question,
                bm25=bm25,
                chunks=chunks,
            )

        except Exception as error:

            print(
                "\n=============================="
            )

            print(
                "ERROR"
            )

            print(
                "=============================="
            )

            print(
                type(error).__name__
            )

            print(
                error
            )

            continue

        # ----------------------------------------------------
        # Final answer
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "             FINAL RAG ANSWER"
        )

        print(
            "========================================"
        )

        print(
            answer
        )

        print(
            "\n========================================"
        )