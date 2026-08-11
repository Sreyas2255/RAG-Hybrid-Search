import re

from rank_bm25 import BM25Okapi

from src.ingestion.pdf_loader import load_pdfs
from src.chunking.chunker import (
    chunk_documents,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


PDF_PATHS = [
    "data/raw/Building_Machine_Learning_Systems.pdf",
    "data/raw/Deep_Learning.pdf",
]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized lexical tokens.

    Examples:

        "What is Deep Learning?"
        ->
        ["what", "is", "deep", "learning"]
    """

    return re.findall(
        r"[a-z0-9]+(?:['-][a-z0-9]+)*",
        text.lower(),
    )


def bm25_tokens(text: str) -> list[str]:
    """
    Create BM25 tokens containing:
        - individual words
        - adjacent word bigrams
    """

    words = [
        word
        for word in tokenize(text)
        if word not in STOP_WORDS
    ]

    # Create bigrams such as:
    #
    # deep learning
    # neural network
    # machine learning
    #
    # represented as:
    #
    # deep__learning
    # neural__network
    # machine__learning

    bigrams = [
        f"{words[i]}__{words[i + 1]}"
        for i in range(len(words) - 1)
    ]

    return words + bigrams


def create_bm25_index():
    """
    Load PDFs, create the same chunks used by the
    dense retrieval pipeline, and build a BM25 index.
    """

    print(
        "=" * 60
    )

    print(
        "BUILDING BM25 INDEX"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------
    # Load PDFs
    # --------------------------------------------------

    print(
        "\nLoading PDFs..."
    )

    documents = load_pdfs(
        PDF_PATHS
    )

    print(
        f"Total pages: {len(documents)}"
    )

    retrievable_pages = [
        document
        for document in documents
        if document.metadata.get(
            "is_retrievable",
            True,
        )
    ]

    print(
        f"Retrievable pages: "
        f"{len(retrievable_pages)}"
    )

    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    print(
        "\nCreating chunks..."
    )

    chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    if not chunks:
        raise RuntimeError(
            "No chunks were created."
        )

    # --------------------------------------------------
    # Make BM25 tokens
    # --------------------------------------------------

    print(
        "\nTokenizing chunks..."
    )

    tokenized_texts = [
        bm25_tokens(
            chunk.page_content
        )
        for chunk in chunks
    ]

    # --------------------------------------------------
    # Check for empty tokenized chunks
    # --------------------------------------------------

    empty_token_count = sum(
        1
        for tokens in tokenized_texts
        if not tokens
    )

    print(
        "Empty tokenized chunks:",
        empty_token_count,
    )

    # --------------------------------------------------
    # Create BM25 index
    # --------------------------------------------------

    print(
        "\nCreating BM25 index..."
    )

    bm25 = BM25Okapi(
        tokenized_texts
    )

    print(
        "BM25 index created successfully."
    )

    return bm25, chunks


def bm25_search(
    bm25,
    chunks,
    query: str,
    top_k: int = 10,
):
    """
    Search the BM25 index using lexical matching.

    Higher BM25 score = stronger lexical match.
    """

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
    # Tokenize query
    # --------------------------------------------------

    query_tokens = bm25_tokens(
        query
    )

    if not query_tokens:
        return []

    # --------------------------------------------------
    # Calculate BM25 scores
    # --------------------------------------------------

    scores = bm25.get_scores(
        query_tokens
    )

    # --------------------------------------------------
    # Get highest-scoring indexes
    # --------------------------------------------------

    top_k = min(
        top_k,
        len(chunks),
    )

    top_indexes = (
        scores.argsort()[
            -top_k:
        ][::-1]
    )

    results = []

    # --------------------------------------------------
    # Build result objects
    # --------------------------------------------------

    for index in top_indexes:

        score = float(
            scores[index]
        )

        # Ignore zero-score chunks.
        if score <= 0:
            continue

        chunk = chunks[index]

        metadata = (
            chunk.metadata.copy()
            if chunk.metadata
            else {}
        )

        # Use the real chunk ID generated
        # by the chunking stage.
        chunk_id = metadata.get(
            "chunk_id",
            f"chunk_{index}",
        )

        results.append(
            {
                "rank": len(results) + 1,
                "id": chunk_id,
                "document": chunk,
                "metadata": metadata,
                "score": score,
            }
        )

    return results


def print_results(
    results: list[dict],
):
    """
    Print BM25 retrieval results.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BM25 RETRIEVAL RESULTS"
    )

    print(
        "=" * 60
    )

    if not results:

        print(
            "\nNo lexical matches found."
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
            "BM25 Score:",
            result["score"],
        )

        print(
            "Source:",
            result["metadata"].get(
                "source",
                "unknown",
            ),
        )

        print(
            "Page:",
            result["metadata"].get(
                "page",
                "unknown",
            ),
        )

        print(
            "Human page:",
            result["metadata"].get(
                "human_page",
                "unknown",
            ),
        )

        print(
            "\nText:"
        )

        print(
            result["document"].page_content[
                :1500
            ]
        )

        print(
            "\n"
            + "-" * 60
        )


if __name__ == "__main__":

    # --------------------------------------------------
    # Build BM25 index
    # --------------------------------------------------

    bm25, chunks = (
        create_bm25_index()
    )

    # --------------------------------------------------
    # Ask for query
    # --------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nERROR: Query cannot be empty."
        )

        raise SystemExit(1)

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=10,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print_results(
        results
    )