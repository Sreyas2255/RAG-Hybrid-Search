from pathlib import Path

import chromadb


CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag_documents"


def create_vector_store(reset: bool = False):
    """
    Create or connect to the persistent ChromaDB collection.

    Parameters
    ----------
    reset:
        If True, delete the existing collection and create a new one.

    Returns
    -------
    chromadb.Collection
        ChromaDB collection.
    """

    # --------------------------------------------------
    # Make sure the Chroma directory exists
    # --------------------------------------------------

    Path(CHROMA_PATH).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create persistent ChromaDB client
    # --------------------------------------------------

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # --------------------------------------------------
    # Reset collection if requested
    # --------------------------------------------------

    if reset:
        print("Resetting ChromaDB collection...")

        try:
            client.delete_collection(
                name=COLLECTION_NAME
            )

            print("Old collection deleted.")

        except Exception:
            print("No existing collection found.")

    # --------------------------------------------------
    # Create or load collection
    # --------------------------------------------------

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    return collection


if __name__ == "__main__":

    collection = create_vector_store()

    print()
    print("ChromaDB connected successfully!")
    print("Collection:", collection.name)
    print(
        "Documents in collection:",
        collection.count(),
    )