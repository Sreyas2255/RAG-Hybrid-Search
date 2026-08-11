from src.ingestion.pdf_loader import load_pdfs

from src.chunking.chunker import (
    chunk_documents,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)

from src.embeddings.embedding import (
    create_embedding_model,
)

from src.retrieval.vector_store import (
    create_vector_store,
)


# --------------------------------------------------
# PDF files
# --------------------------------------------------

PDF_PATHS = [
    "data/raw/Building_Machine_Learning_Systems.pdf",
    "data/raw/Deep_Learning.pdf",
]


# --------------------------------------------------
# Indexing configuration
# --------------------------------------------------

BATCH_SIZE = 64


def main():

    # --------------------------------------------------
    # STEP 1: Load PDFs
    # --------------------------------------------------

    print("=" * 60)
    print("STEP 1: LOADING PDFs")
    print("=" * 60)

    documents = load_pdfs(PDF_PATHS)

    print()
    print(f"Total pages loaded: {len(documents)}")

    retrievable_pages = [
        document
        for document in documents
        if document.metadata.get(
            "is_retrievable",
            True,
        )
    ]

    print(
        f"Retrievable pages: {len(retrievable_pages)}"
    )


    # --------------------------------------------------
    # STEP 2: Create chunks
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 2: CREATING CHUNKS")
    print("=" * 60)

    chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )


    # --------------------------------------------------
    # STEP 3: Load embedding model
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 3: LOADING EMBEDDING MODEL")
    print("=" * 60)

    embedding_model = create_embedding_model()

    print(
        "Embedding model loaded successfully."
    )


    # --------------------------------------------------
    # STEP 4: Create/reset ChromaDB
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 4: CREATING CHROMADB COLLECTION")
    print("=" * 60)

    collection = create_vector_store(
        reset=True
    )

    print(
        "ChromaDB collection ready:"
    )

    print(
        f"Collection name: {collection.name}"
    )


    # --------------------------------------------------
    # STEP 5: Prepare valid chunks
    # --------------------------------------------------

    valid_chunks = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if not text:
            continue

        valid_chunks.append(chunk)

    print()
    print(
        f"Valid chunks to index: {len(valid_chunks)}"
    )


    # --------------------------------------------------
    # STEP 6: Create embeddings in batches
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("STEP 5: CREATING EMBEDDINGS")
    print("=" * 60)

    total_chunks = len(valid_chunks)

    for start in range(
        0,
        total_chunks,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total_chunks,
        )

        batch = valid_chunks[start:end]

        texts = [
            chunk.page_content.strip()
            for chunk in batch
        ]

        # --------------------------------------------------
        # Use embed_documents() for document indexing.
        # --------------------------------------------------

        embeddings = (
            embedding_model.embed_documents(
                texts
            )
        )

        # --------------------------------------------------
        # Prepare metadata
        # --------------------------------------------------

        metadatas = []
        ids = []

        for chunk in batch:

            metadata = {
                key: value
                for key, value
                in chunk.metadata.items()
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                )
            }

            # Preserve the chunk ID generated
            # by the chunking stage.

            chunk_id = metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                chunk_id = (
                    f"chunk_{start + len(ids):06d}"
                )

            metadata["chunk_id"] = str(
                chunk_id
            )

            metadata[
                "is_retrievable"
            ] = True

            metadatas.append(
                metadata
            )

            ids.append(
                str(chunk_id)
            )

        # --------------------------------------------------
        # Store batch in ChromaDB
        # --------------------------------------------------

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Indexed {end}/{total_chunks} "
            f"chunks"
        )


    # --------------------------------------------------
    # STEP 7: Verify index
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)

    count = collection.count()

    print(
        f"Documents in ChromaDB: {count}"
    )

    print(
        f"Expected documents: {total_chunks}"
    )

    if count == total_chunks:

        print(
            "SUCCESS: All chunks were indexed."
        )

    else:

        print(
            "WARNING: Indexed count does not "
            "match expected count."
        )


if __name__ == "__main__":
    main()