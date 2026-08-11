from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def _retrievable_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Return only documents marked as retrievable.
    """

    return [
        document
        for document in documents
        if document.metadata.get(
            "is_retrievable",
            True,
        )
    ]


def _create_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Create the recursive text splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def _create_fixed_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Create a fixed-size character splitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[""],
    )


def _process_pages(
    documents: list[Document],
    splitter: RecursiveCharacterTextSplitter,
    strategy: str,
) -> list[Document]:
    """
    Split each PDF page independently.

    This is important because we do not want a chunk
    to contain text from two different PDF pages.

    Each resulting chunk keeps the metadata of its
    original page.
    """

    all_chunks: list[Document] = []

    for document in documents:

        text = document.page_content.strip()

        # Ignore completely empty pages.
        if not text:
            continue

        page_chunks = splitter.split_text(text)

        if not page_chunks:
            continue

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            0,
        )

        human_page = document.metadata.get(
            "human_page",
            page + 1,
        )

        total_chunks_in_page = len(
            page_chunks
        )

        for chunk_index, chunk_text in enumerate(
            page_chunks
        ):

            # Create a stable readable ID.
            #
            # Example:
            # Building_Machine_Learning_Systems.pdf
            # page 99
            # chunk 2
            #
            # becomes:
            #
            # chunk_0099_002
            #
            chunk_id = (
                f"chunk_"
                f"{page:04d}_"
                f"{chunk_index:03d}_"
                f"{len(all_chunks):06d}"
            )

            # Copy original metadata.
            metadata = dict(
                document.metadata
            )

            # Add chunk metadata.
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "total_chunks_in_page": (
                        total_chunks_in_page
                    ),
                    "chunking_strategy": strategy,
                    "character_count": len(
                        chunk_text
                    ),
                    "is_retrievable": True,
                    "source": source,
                    "page": page,
                    "human_page": human_page,
                }
            )

            chunk = Document(
                page_content=chunk_text,
                metadata=metadata,
            )

            all_chunks.append(chunk)

    return all_chunks


def recursive_chunking(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split retrievable PDF pages into recursive chunks.

    Each page is processed independently.
    """

    retrievable_documents = (
        _retrievable_documents(documents)
    )

    splitter = _create_recursive_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return _process_pages(
        documents=retrievable_documents,
        splitter=splitter,
        strategy="recursive",
    )


def fixed_size_chunking(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split retrievable PDF pages into fixed-size chunks.

    Each page is processed independently.
    """

    retrievable_documents = (
        _retrievable_documents(documents)
    )

    splitter = _create_fixed_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return _process_pages(
        documents=retrievable_documents,
        splitter=splitter,
        strategy="fixed",
    )


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Select the requested chunking strategy.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size."
        )

    if strategy == "recursive":

        return recursive_chunking(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if strategy == "fixed":

        return fixed_size_chunking(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    raise ValueError(
        f"Unknown chunking strategy: {strategy}. "
        f"Use 'recursive' or 'fixed'."
    )


def print_chunk_statistics(
    documents: list[Document],
    chunks: list[Document],
) -> None:
    """
    Print useful chunking statistics.
    """

    print()
    print("=" * 60)
    print("CHUNKING STATISTICS")
    print("=" * 60)

    print(
        f"Input pages:          {len(documents)}"
    )

    print(
        f"Output chunks:        {len(chunks)}"
    )

    if documents:

        average_chunks = (
            len(chunks) / len(documents)
        )

        print(
            f"Average chunks/page:  "
            f"{average_chunks:.2f}"
        )

    if chunks:

        lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        print(
            f"Smallest chunk:       "
            f"{min(lengths)} characters"
        )

        print(
            f"Largest chunk:        "
            f"{max(lengths)} characters"
        )

        print(
            f"Average chunk:        "
            f"{sum(lengths) / len(lengths):.2f} characters"
        )

        print(
            f"Chunk size setting:   "
            f"{DEFAULT_CHUNK_SIZE}"
        )

        print(
            f"Chunk overlap:        "
            f"{DEFAULT_CHUNK_OVERLAP}"
        )


if __name__ == "__main__":

    # Import the PDF loader from Step 1.
    from src.ingestion.pdf_loader import (
        load_pdfs,
    )

    # --------------------------------------------------------
    # PDF files
    # --------------------------------------------------------

    pdf_paths = [
        "data/raw/Building_Machine_Learning_Systems.pdf",
        "data/raw/Deep_Learning.pdf",
    ]

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    documents = load_pdfs(
        pdf_paths
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = chunk_documents(
        documents,
        strategy="recursive",
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print_chunk_statistics(
        documents,
        chunks,
    )

    # --------------------------------------------------------
    # Show first chunk
    # --------------------------------------------------------

    if chunks:

        print()
        print("=" * 60)
        print("FIRST CHUNK")
        print("=" * 60)

        print(
            chunks[0].page_content
        )

        print()
        print("Metadata:")

        print(
            chunks[0].metadata
        )

    # --------------------------------------------------------
    # Show first five chunks
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)

    for chunk in chunks[:5]:

        print()
        print(
            f"Chunk ID: "
            f"{chunk.metadata.get('chunk_id')}"
        )

        print(
            f"Source: "
            f"{chunk.metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{chunk.metadata.get('human_page')}"
        )

        print(
            f"Chunk index: "
            f"{chunk.metadata.get('chunk_index')}"
        )

        print(
            f"Characters: "
            f"{chunk.metadata.get('character_count')}"
        )

        print(
            f"Text:"
        )

        print(
            chunk.page_content[:500]
        )

        print("-" * 60)