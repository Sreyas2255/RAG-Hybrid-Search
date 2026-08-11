from src.retrieval.vector_store import create_vector_store

collection = create_vector_store()

print("Documents:", collection.count())

results = collection.get(
    limit=20,
    include=["documents", "metadatas"],
)

print("\n==============================")
print("SAMPLE CHROMA DOCUMENTS")
print("==============================")

for i, (doc, metadata) in enumerate(
    zip(
        results["documents"],
        results["metadatas"],
    ),
    start=1,
):
    print(f"\n--- {i} ---")
    print("Source:", metadata.get("source"))
    print("Page:", metadata.get("page"))
    print(doc[:500])