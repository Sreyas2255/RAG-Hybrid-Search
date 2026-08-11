from src.ingestion.pdf_loader import load_pdfs
from src.chunking.chunker import chunk_documents


PDF_PATHS = [
    "data/raw/Building_Machine_Learning_Systems.pdf",
    "data/raw/Deep_Learning.pdf",
]


def create_bm25_index():

    print("Loading PDFs...")

    documents = load_pdfs(PDF_PATHS)

    print(f"Total pages: {len(documents)}")

    print("\nCreating chunks...")

    chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Total chunks: {len(chunks)}")

    from rank_bm25 import BM25Okapi

    # Extract text from chunks
    texts = [chunk.page_content for chunk in chunks]

    # Tokenize
    tokenized_texts = [
        text.lower().split()
        for text in texts
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_texts)

    return bm25, chunks


def bm25_search(bm25, chunks, query, top_k=5):

    # Tokenize user query
    query_tokens = query.lower().split()

    # Search
    scores = bm25.get_scores(query_tokens)

    # Get highest scoring indexes
    top_indexes = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indexes:

        results.append(
            {
                "id": f"chunk_{index}",
                "document": chunks[index],
                "metadata": chunks[index].metadata,
                "score": scores[index],
            }
        )

    return results


if __name__ == "__main__":

    bm25, chunks = create_bm25_index()

    query = input("\nEnter your question: ")

    results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=5,
    )

    print("\n==============================")
    print("BM25 RETRIEVAL RESULTS")
    print("==============================")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")

        print("ID:", result["id"])

        print("Score:", result["score"])

        print("Source:", result["metadata"].get("source"))

        print("Page:", result["metadata"].get("page"))

        print("\nText:")

        print(result["document"].page_content[:500])