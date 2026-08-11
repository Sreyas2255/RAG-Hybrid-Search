from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_retriever import hybrid_search


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_reranker():
    """
    Create the cross-encoder reranker.
    """

    print("Loading reranker model...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print("Reranker model loaded successfully.")

    return reranker


def rerank_results(
    query: str,
    results: list,
    top_k: int = 5,
):
    """
    Rerank hybrid retrieval results using
    a cross-encoder model.
    """

    if not results:
        return []

    reranker = create_reranker()

    # --------------------------------------------------
    # Create (query, document) pairs
    # --------------------------------------------------

    pairs = [
        (
            query,
            result["document"].page_content
            if hasattr(
                result["document"],
                "page_content",
            )
            else result["document"],
        )
        for result in results
    ]

    print(
        f"Reranking {len(pairs)} candidates..."
    )

    # --------------------------------------------------
    # Calculate reranker scores
    # --------------------------------------------------

    scores = reranker.predict(
        pairs
    )

    # --------------------------------------------------
    # Attach reranker scores
    # --------------------------------------------------

    reranked_results = []

    for result, score in zip(
        results,
        scores,
    ):

        reranked_result = result.copy()

        reranked_result[
            "reranker_score"
        ] = float(score)

        reranked_results.append(
            reranked_result
        )

    # --------------------------------------------------
    # Sort by reranker score
    # --------------------------------------------------

    reranked_results.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True,
    )

    # --------------------------------------------------
    # Return top results
    # --------------------------------------------------

    return reranked_results[:top_k]


def main():

    print(
        "\n=============================="
    )

    print(
        "RERANKER TEST"
    )

    print(
        "=============================="
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "Please enter a question."
        )

        return

    # --------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------

    print(
        "\nRunning hybrid retrieval..."
    )

    results = hybrid_search(
        query,
        top_k=30,
    )

    print(
        f"Hybrid results: {len(results)}"
    )

    # --------------------------------------------------
    # Reranking
    # --------------------------------------------------

    results = rerank_results(
        query,
        results,
        top_k=5,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

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
        results,
        start=1,
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            "ID:",
            result["id"],
        )

        print(
            "Reranker Score:",
            result[
                "reranker_score"
            ],
        )

        print(
            "RRF Score:",
            result.get(
                "rrf_score"
            ),
        )

        print(
            "Source:",
            result[
                "metadata"
            ].get(
                "source"
            ),
        )

        print(
            "Page:",
            result[
                "metadata"
            ].get(
                "page"
            ),
        )

        print(
            "Human Page:",
            result[
                "metadata"
            ].get(
                "human_page"
            ),
        )

        print(
            "\nText:"
        )

        document = result[
            "document"
        ]

        if hasattr(
            document,
            "page_content",
        ):
            text = (
                document.page_content
            )
        else:
            text = document

        print(
            text[:1200]
        )


if __name__ == "__main__":
    main()