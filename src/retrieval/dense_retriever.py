from src.embeddings.embedding import (
    create_embedding_model,
)

from src.retrieval.vector_store import (
    create_vector_store,
)


def dense_search(
    query: str,
    top_k: int = 10,
):
    """
    Perform semantic search using ChromaDB.

    Workflow:
        1. Load the embedding model.
        2. Connect to ChromaDB.
        3. Convert the query into an embedding.
        4. Search ChromaDB using the query embedding.
        5. Return the closest chunks.
    """

    # --------------------------------------------------
    # Validate query
    # --------------------------------------------------

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print(
        "Loading embedding model..."
    )

    embedding_model = (
        create_embedding_model()
    )

    # --------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------

    print(
        "Connecting to ChromaDB..."
    )

    collection = (
        create_vector_store()
    )

    document_count = (
        collection.count()
    )

    if document_count == 0:

        raise RuntimeError(
            "ChromaDB collection is empty.\n"
            "Run:\n"
            "python -m src.retrieval.index_documents"
        )

    print(
        f"Documents in collection: "
        f"{document_count}"
    )

    # --------------------------------------------------
    # Limit top_k
    # --------------------------------------------------

    top_k = min(
        top_k,
        document_count,
    )

    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    print(
        "Creating query embedding..."
    )

    query_embedding = (
        embedding_model.embed_query(
            query
        )
    )

    # --------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------

    print(
        "Searching ChromaDB..."
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # --------------------------------------------------
    # Extract results
    # --------------------------------------------------

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    ids = results.get(
        "ids",
        [[]],
    )[0]

    # --------------------------------------------------
    # Build result objects
    # --------------------------------------------------

    retrieved_results = []

    for rank, (
        document_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        metadata = metadata or {}

        retrieved_results.append(
            {
                "rank": rank,
                "id": document_id,
                "document": document,
                "metadata": metadata,
                "distance": float(
                    distance
                ),
            }
        )

    return retrieved_results


def print_results(
    results: list[dict],
):
    """
    Print dense retrieval results.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DENSE RETRIEVAL RESULTS"
    )

    print(
        "=" * 60
    )

    if not results:

        print(
            "\nNo results found."
        )

        return

    for result in results:

        print(
            f"\n--- Result "
            f"{result['rank']} ---"
        )

        print(
            "ID:",
            result["id"],
        )

        print(
            "Distance:",
            result["distance"],
        )

        metadata = (
            result["metadata"]
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "unknown",
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "unknown",
            ),
        )

        print(
            "Human page:",
            metadata.get(
                "human_page",
                "unknown",
            ),
        )

        print(
            "\nText:"
        )

        print(
            result["document"][:1500]
        )

        print(
            "\n"
            + "-" * 60
        )


if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "DENSE RETRIEVER TEST"
    )

    print(
        "=" * 60
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nERROR: Query cannot be empty."
        )

        raise SystemExit(1)

    results = dense_search(
        query=query,
        top_k=10,
    )

    print_results(
        results
    )