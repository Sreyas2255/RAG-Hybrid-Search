from src.retrieval.dense_retriever import dense_search
from src.retrieval.bm25_retriever import (
    create_bm25_index,
    bm25_search,
)


# ============================================================
# HYBRID RETRIEVER CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 10

# Number of candidates retrieved from each retriever
DENSE_TOP_K = 30
BM25_TOP_K = 30

# RRF constant
# Standard RRF commonly uses k = 60.
RRF_K = 60


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Combine Dense and BM25 retrieval results using
    Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF score = 1 / (k + rank)

    A document receives a score from each retriever
    where it appears.

    Documents appearing highly in both retrievers
    receive the strongest combined score.
    """

    fused = {}

    # ========================================================
    # ADD DENSE RESULTS
    # ========================================================

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        document_id = result["id"]

        if document_id not in fused:

            fused[document_id] = {
                "id": document_id,
                "document": result["document"],
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }

        fused[document_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[document_id]["dense_rank"] = rank

        fused[document_id]["dense_score"] = (
            result.get("score")
        )

    # ========================================================
    # ADD BM25 RESULTS
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        document = result["document"]

        # ----------------------------------------------------
        # BM25 currently returns the original LangChain
        # Document object.
        #
        # Dense retrieval returns a Chroma ID.
        #
        # We therefore use the chunk_id metadata when
        # available so both retrievers can identify the
        # same chunk.
        # ----------------------------------------------------

        metadata = getattr(
            document,
            "metadata",
            {},
        )

        document_id = metadata.get(
            "chunk_id"
        )

        # ----------------------------------------------------
        # Fallback ID
        # ----------------------------------------------------

        if document_id is None:

            document_id = result["id"]

        # ----------------------------------------------------
        # Make sure the ID is a string.
        # ----------------------------------------------------

        document_id = str(document_id)

        if document_id not in fused:

            fused[document_id] = {
                "id": document_id,
                "document": document,
                "metadata": metadata,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            }

        fused[document_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

        fused[document_id]["bm25_rank"] = rank

        fused[document_id]["bm25_score"] = (
            result.get("score")
        )

    # ========================================================
    # SORT BY RRF SCORE
    # ========================================================

    fused_results = list(
        fused.values()
    )

    fused_results.sort(
        key=lambda result: result["rrf_score"],
        reverse=True,
    )

    # ========================================================
    # ADD FINAL RANK
    # ========================================================

    for rank, result in enumerate(
        fused_results,
        start=1,
    ):

        result["rank"] = rank

        # ----------------------------------------------------
        # Track which retrievers found this document.
        # ----------------------------------------------------

        if (
            result["dense_rank"] is not None
            and result["bm25_rank"] is not None
        ):

            result["retrieval_source"] = (
                "dense+bm25"
            )

        elif result["dense_rank"] is not None:

            result["retrieval_source"] = (
                "dense"
            )

        else:

            result["retrieval_source"] = (
                "bm25"
            )

    return fused_results[:top_k]


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    dense_top_k: int = DENSE_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Perform hybrid retrieval using:

        Dense semantic search
        +
        BM25 lexical search
        +
        Reciprocal Rank Fusion

    Workflow:

        Query
          |
          +----> Dense Retriever
          |
          +----> BM25 Retriever
          |
          v
        RRF Fusion
          |
          v
        Deduplicated ranking
          |
          v
        Top-K results
    """

    # ========================================================
    # VALIDATE QUERY
    # ========================================================

    query = query.strip()

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )

    # ========================================================
    # DENSE RETRIEVAL
    # ========================================================

    print(
        "\nRunning dense retrieval..."
    )

    dense_results = dense_search(
        query,
        top_k=dense_top_k,
    )

    print(
        f"Dense results: {len(dense_results)}"
    )

    # ========================================================
    # BM25 INDEX
    # ========================================================

    print(
        "\nBuilding BM25 index..."
    )

    bm25, chunks = create_bm25_index()

    # ========================================================
    # BM25 RETRIEVAL
    # ========================================================

    print(
        "\nRunning BM25 retrieval..."
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=bm25_top_k,
    )

    print(
        f"BM25 results: {len(bm25_results)}"
    )

    # ========================================================
    # RRF FUSION
    # ========================================================

    print(
        "\nCombining results with RRF..."
    )

    results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results: list[dict],
) -> None:
    """
    Print hybrid retrieval results.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HYBRID RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

    for result in results:

        print(
            f"\n--- Result {result['rank']} ---"
        )

        print(
            "ID:",
            result["id"],
        )

        print(
            "RRF Score:",
            round(
                result["rrf_score"],
                6,
            ),
        )

        print(
            "Retrieval Source:",
            result["retrieval_source"],
        )

        print(
            "Dense Rank:",
            result["dense_rank"],
        )

        print(
            "BM25 Rank:",
            result["bm25_rank"],
        )

        print(
            "Dense Score:",
            result["dense_score"],
        )

        print(
            "BM25 Score:",
            result["bm25_score"],
        )

        print(
            "Source:",
            result["metadata"].get(
                "source"
            ),
        )

        print(
            "Page:",
            result["metadata"].get(
                "page"
            ),
        )

        print(
            "Human Page:",
            result["metadata"].get(
                "human_page"
            ),
        )

        print(
            "\nText:"
        )

        document = result["document"]

        # ----------------------------------------------------
        # Dense results contain a string.
        #
        # BM25 results contain a LangChain Document.
        # ----------------------------------------------------

        if hasattr(
            document,
            "page_content",
        ):

            text = document.page_content

        else:

            text = str(document)

        print(
            text[:1000]
        )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HYBRID RETRIEVER TEST"
    )

    print(
        "=" * 70
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "Please enter a question."
        )

        return

    results = hybrid_search(
        query=query,
        top_k=10,
        dense_top_k=30,
        bm25_top_k=30,
        rrf_k=60,
    )

    print_results(
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()