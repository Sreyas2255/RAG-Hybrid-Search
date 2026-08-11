from src.retrieval.vector_store import (
    create_embedding_model,
    create_vector_store,
)


def dense_search(query: str, top_k: int = 5):
    """
    Search ChromaDB using semantic similarity.
    """

    # Load embedding model
    embedding_model = create_embedding_model()

    # Connect to ChromaDB
    collection = create_vector_store()

    # Convert user's question into an embedding
    query_embedding = embedding_model.embed_query(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results


if __name__ == "__main__":

    query = input("Enter your question: ")

    results = dense_search(query, top_k=5)

    print("\n==============================")
    print("DENSE RETRIEVAL RESULTS")
    print("==============================")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):

        print(f"\n--- Result {i} ---")

        print("Distance:", distance)

        print("Source:", metadata.get("source"))

        print("Page:", metadata.get("page"))

        print("\nText:")
        print(document[:500])