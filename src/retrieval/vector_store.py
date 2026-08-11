from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_PATH = "data/chroma"


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store():
    """
    Create a persistent ChromaDB client and collection.
    """

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="rag_documents"
    )

    return collection


if __name__ == "__main__":

    embedding_model = create_embedding_model()

    collection = create_vector_store()

    print("ChromaDB connected successfully!")
    print("Collection:", collection.name)
    print("Documents in collection:", collection.count())