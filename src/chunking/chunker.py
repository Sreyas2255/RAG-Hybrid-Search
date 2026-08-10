from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document


def fixed_size_chunking(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents into fixed-size chunks with overlap.
    """

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunking_strategy"] = "fixed"
        chunk.metadata["character_count"] = len(chunk.page_content)

    return chunks


def recursive_chunking(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents using recursive/structure-aware chunking.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunking_strategy"] = "recursive"
        chunk.metadata["character_count"] = len(chunk.page_content)

    return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Select the chunking strategy.
    """

    if strategy == "fixed":
        return fixed_size_chunking(
            documents,
            chunk_size,
            chunk_overlap,
        )

    elif strategy == "recursive":
        return recursive_chunking(
            documents,
            chunk_size,
            chunk_overlap,
        )

    else:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )