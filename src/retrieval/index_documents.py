from src.ingestion.pdf_loader import load_pdfs
from src.chunking.chunker import chunk_documents
from src.retrieval.vector_store import (
    create_embedding_model,
    create_vector_store,
)


PDF_PATHS = [
    "data/raw/Building_Machine_Learning_Systems.pdf",
    "data/raw/Deep_Learning.pdf",
]


def main():

    # 1. Load PDFs
    print("Loading PDFs...")

    documents = load_pdfs(PDF_PATHS)

    print(f"Total pages: {len(documents)}")

    # 2. Create chunks
    print("\nCreating chunks...")

    chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Total chunks: {len(chunks)}")

    # 3. Load embedding model
    print("\nLoading embedding model...")

    embedding_model = create_embedding_model()

    # 4. Connect to ChromaDB
    print("\nConnecting to ChromaDB...")

    collection = create_vector_store()

    # 5. Prepare data
    texts = []
    embeddings = []
    metadatas = []
    ids = []

    print("\nCreating embeddings...")

    for i, chunk in enumerate(chunks):

        text = chunk.page_content

        embedding = embedding_model.embed_query(text)

        metadata = chunk.metadata.copy()

        # Chroma metadata values must be simple types
        metadata["chunk_id"] = i

        texts.append(text)
        embeddings.append(embedding)
        metadatas.append(metadata)
        ids.append(f"chunk_{i}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")

    # 6. Insert into ChromaDB
    print("\nStoring chunks in ChromaDB...")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("\nIndexing completed!")
    print("Documents in ChromaDB:", collection.count())


if __name__ == "__main__":
    main()