from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_paths: list[str]):
    """Load multiple PDF files and return all documents."""

    all_documents = []

    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        all_documents.extend(documents)

        print(f"Pages loaded: {len(documents)}")

    return all_documents


if __name__ == "__main__":

    pdf_paths = [
        "data/raw/Building_Machine_Learning_Systems.pdf",
        "data/raw/Deep_Learning.pdf",
    ]

    documents = load_pdfs(pdf_paths)

    print("\n==============================")
    print("TOTAL DOCUMENTS/PAGES LOADED")
    print("==============================")

    print(f"Total pages: {len(documents)}")

    print("\n==============================")
    print("FIRST DOCUMENT")
    print("==============================")

    print(documents[0].page_content[:1000])

    print("\nMetadata:")

    print(documents[0].metadata)