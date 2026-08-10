from langchain_community.document_loaders import PyPDFLoader

from src.chunking.chunker import chunk_documents


pdf_paths = [
    "data/raw/Building_Machine_Learning_Systems.pdf",
    "data/raw/Deep_Learning.pdf",
]


# -----------------------------
# 1. Load both PDFs
# -----------------------------

documents = []

for pdf_path in pdf_paths:

    print(f"\nLoading: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    documents.extend(docs)

    print(f"Pages loaded: {len(docs)}")


# -----------------------------
# 2. Check total pages
# -----------------------------

print("\n==============================")
print("INGESTION SUMMARY")
print("==============================")

print(f"Total pages: {len(documents)}")


# -----------------------------
# 3. Select chunking strategy
# -----------------------------

strategy = "fixed"

chunks = chunk_documents(
    documents,
    strategy=strategy,
    chunk_size=1000,
    chunk_overlap=200,
)


# -----------------------------
# 4. Check chunks
# -----------------------------

print("\n==============================")
print("CHUNKING SUMMARY")
print("==============================")

print(f"Strategy: {strategy}")
print(f"Total chunks: {len(chunks)}")


# -----------------------------
# 5. Inspect first 5 chunks
# -----------------------------

for i, chunk in enumerate(chunks[:5]):

    print("\n==============================")
    print(f"CHUNK {i}")
    print("==============================")

    print(chunk.page_content[:500])

    print("\nMetadata:")
    print(chunk.metadata)