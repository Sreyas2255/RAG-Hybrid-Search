from src.retrieval.dense_retriever import dense_search
from src.retrieval.bm25_retriever import create_bm25_index, bm25_search


def normalize_result(result):
    """
    Convert Dense/BM25 results into one common format.
    """

    document = result.get("document")
    metadata = result.get("metadata", {})

    # If document is a LangChain Document
    if hasattr(document, "page_content"):
        text = document.page_content

        # Use Document metadata if result metadata is empty
        if not metadata:
            metadata = document.metadata

    # If document is already a string
    else:
        text = document

    return {
        "text": text,
        "metadata": metadata,
    }


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    top_k=5,
):
    """
    Combine Dense and BM25 results using
    Reciprocal Rank Fusion (RRF).
    """

    scores = {}
    documents = {}

    # ==========================================
    # Dense Retrieval Results
    # ==========================================

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        document_id = result["id"]

        normalized = normalize_result(result)

        scores[document_id] = scores.get(
            document_id,
            0,
        )

        scores[document_id] += 1 / (k + rank)

        documents[document_id] = normalized

    # ==========================================
    # BM25 Retrieval Results
    # ==========================================

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        document_id = result["id"]

        normalized = normalize_result(result)

        scores[document_id] = scores.get(
            document_id,
            0,
        )

        scores[document_id] += 1 / (k + rank)

        documents[document_id] = normalized

    # ==========================================
    # Sort by RRF score
    # ==========================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for document_id, score in ranked[:top_k]:

        results.append(
            {
                "id": document_id,
                "text": documents[document_id]["text"],
                "metadata": documents[document_id]["metadata"],
                "score": score,
            }
        )

    return results


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------
    # Create BM25 index
    # ------------------------------------------

    print("Creating BM25 index...")

    bm25, chunks = create_bm25_index()

    # ------------------------------------------
    # Get user question
    # ------------------------------------------

    query = input("\nEnter your question: ")

    # ------------------------------------------
    # Dense Retrieval
    # ------------------------------------------

    print("\nRunning Dense Retrieval...")

    dense_results = dense_search(
        query,
        top_k=10,
    )

    # ------------------------------------------
    # BM25 Retrieval
    # ------------------------------------------

    print("\nRunning BM25 Retrieval...")

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=10,
    )

    # ------------------------------------------
    # Hybrid Retrieval
    # ------------------------------------------

    print("\nCombining results using RRF...")

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        k=60,
        top_k=5,
    )

    # ------------------------------------------
    # Display results
    # ------------------------------------------

    print("\n==============================")
    print("HYBRID RETRIEVAL RESULTS")
    print("==============================")

    for i, result in enumerate(
        hybrid_results,
        start=1,
    ):

        print(f"\n--- Result {i} ---")

        print("ID:", result["id"])

        print("RRF Score:", result["score"])

        metadata = result["metadata"]

        print(
            "Source:",
            metadata.get("source"),
        )

        print(
            "Page:",
            metadata.get("page"),
        )

        print("\nText:")

        print(
            result["text"][:500]
        )