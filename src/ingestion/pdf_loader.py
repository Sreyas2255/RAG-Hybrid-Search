from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDFs.

    The PDF files used in this project sometimes contain:
    - missing spaces between words
    - broken words across lines
    - excessive whitespace
    - spaces around punctuation

    This function performs conservative cleaning without
    aggressively changing the original content.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # --------------------------------------------------------
    # Fix words broken across lines
    #
    # Example:
    #
    # applica-
    # tion
    #
    # becomes:
    #
    # application
    # --------------------------------------------------------

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    # --------------------------------------------------------
    # Replace newlines with spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s*\n\s*",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Collapse repeated whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Fix common PDF extraction fragments
    # --------------------------------------------------------

    replacements = {
        "a nd": "and",
        "a re": "are",
        "a bout": "about",
        "a s": "as",
        "a t": "at",
        "a n": "an",
        "a ll": "all",
        "a lso": "also",
        "a ny": "any",
        "a way": "away",

        "b ecause": "because",
        "b etween": "between",

        "c an": "can",
        "c onsists": "consists",
        "c ontains": "contains",

        "d ata": "data",
        "d eep": "deep",
        "d oes": "does",
        "d ifferent": "different",

        "e ach": "each",
        "e ven": "even",

        "f or": "for",
        "f rom": "from",

        "h as": "has",
        "h ave": "have",
        "h ad": "had",

        "i n": "in",
        "i nto": "into",
        "i s": "is",
        "i t": "it",

        "l earning": "learning",
        "l ayer": "layer",
        "l ayers": "layers",

        "m achine": "machine",
        "m ay": "may",
        "m odel": "model",
        "m odels": "models",
        "m ore": "more",

        "n etwork": "network",
        "n etworks": "networks",

        "o f": "of",
        "o ne": "one",
        "o utput": "output",

        "p rediction": "prediction",
        "p redictions": "predictions",

        "r ecognition": "recognition",

        "s ince": "since",
        "s ystem": "system",
        "s ystems": "systems",

        "t he": "the",
        "t han": "than",
        "t hat": "that",
        "t heir": "their",
        "t hese": "these",
        "t his": "this",
        "t o": "to",
        "t oday": "today",
        "t ogether": "together",

        "u sing": "using",

        "w as": "was",
        "w ere": "were",
        "w ith": "with",
        "w ithin": "within",
    }

    for broken, correct in replacements.items():
        text = re.sub(
            rf"\b{re.escape(broken)}\b",
            correct,
            text,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Fix common broken words
    # --------------------------------------------------------

    common_broken_words = {
        "introducti on": "introduction",
        "applicati on": "application",
        "recogniti on": "recognition",
        "predicti on": "prediction",
        "classificati on": "classification",
        "computati on": "computation",
        "informati on": "information",
        "representati on": "representation",
        "organizati on": "organization",
        "normalizati on": "normalization",
        "regularizati on": "regularization",
        "generalizati on": "generalization",
        "connecti on": "connection",
        "collecti on": "collection",
        "functi on": "function",

        "transformat ion": "transformation",
        "transformatio n": "transformation",

        "architectu re": "architecture",
        "architect ure": "architecture",

        "artifici al": "artificial",
        "neura l": "neural",

        "ne twork": "network",
        "networ k": "network",

        "learn ing": "learning",
        "train ing": "training",
        "process ing": "processing",

        "comput er": "computer",
        "soft ware": "software",
        "hard ware": "hardware",

        "dat a": "data",

        "algorit hm": "algorithm",
        "algorith m": "algorithm",

        "paramet er": "parameter",
        "paramet ers": "parameters",

        "activati on": "activation",
        "activ ation": "activation",

        "laye r": "layer",
        "laye rs": "layers",

        "input s": "inputs",
        "outpu t": "output",
        "outpu ts": "outputs",
    }

    for broken, correct in common_broken_words.items():
        text = re.sub(
            rf"\b{re.escape(broken)}\b",
            correct,
            text,
            flags=re.IGNORECASE,
        )

    # --------------------------------------------------------
    # Fix common chapter headings
    #
    # Example:
    #
    # CHAPTER1. INTRODUCTION
    #
    # becomes:
    #
    # CHAPTER 1. INTRODUCTION
    # --------------------------------------------------------

    text = re.sub(
        r"\bCHAPTER\s*(\d+)\s*\.\s*",
        r"CHAPTER \1. ",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # Fix punctuation spacing
    # --------------------------------------------------------

    text = re.sub(
        r"\s+([,.!?;:%])",
        r"\1",
        text,
    )

    text = re.sub(
        r"([(\[])\s+",
        r"\1",
        text,
    )

    text = re.sub(
        r"\s+([)\]])",
        r"\1",
        text,
    )

    # --------------------------------------------------------
    # Final whitespace cleanup
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# PDF LOADING
# ============================================================

def load_pdfs(
    pdf_paths: list[str],
) -> list[Document]:
    """
    Load multiple PDF files using PyMuPDF.

    IMPORTANT:
    Every PDF page is returned as a LangChain Document.

    This is required because the chunking stage expects:

        document.page_content
        document.metadata
    """

    all_documents: list[Document] = []

    for pdf_path in pdf_paths:

        print(
            f"Loading: {pdf_path}"
        )

        # ----------------------------------------------------
        # Check that the file exists
        # ----------------------------------------------------

        path = Path(pdf_path)

        if not path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        # ----------------------------------------------------
        # Open PDF
        # ----------------------------------------------------

        pdf = pymupdf.open(
            str(path)
        )

        total_pages = len(pdf)

        print(
            f"Pages loaded: {total_pages}"
        )

        retrievable_count = 0

        # ----------------------------------------------------
        # Process every page independently
        # ----------------------------------------------------

        for page_number in range(
            total_pages
        ):

            page = pdf[page_number]

            # Extract text.
            raw_text = page.get_text(
                "text"
            )

            # Clean extracted text.
            cleaned_text = clean_pdf_text(
                raw_text
            )

            # Determine whether page has usable text.
            is_retrievable = bool(
                cleaned_text.strip()
            )

            if is_retrievable:
                retrievable_count += 1

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "source": str(path),
                "page": page_number,
                "human_page": page_number + 1,
                "total_pages": total_pages,
                "is_retrievable": is_retrievable,
            }

            # ------------------------------------------------
            # Create a REAL LangChain Document
            #
            # This is the important fix.
            # ------------------------------------------------

            document = Document(
                page_content=cleaned_text,
                metadata=metadata,
            )

            all_documents.append(
                document
            )

        # ----------------------------------------------------
        # Close PDF
        # ----------------------------------------------------

        pdf.close()

        print(
            f"Retrievable pages: "
            f"{retrievable_count}"
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_retrievable = sum(
        1
        for document in all_documents
        if document.metadata.get(
            "is_retrievable",
            False,
        )
    )

    print()
    print(
        f"Total pages: {len(all_documents)}"
    )

    print(
        f"Retrievable pages: "
        f"{total_retrievable}"
    )

    return all_documents


# ============================================================
# TEST / COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # PDF files used by the project
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
    # FIRST DOCUMENT TEST
    # --------------------------------------------------------

    print()
    print(
        "=============================="
    )
    print(
        "FIRST DOCUMENT"
    )
    print(
        "=============================="
    )

    if documents:

        first_document = documents[0]

        print(
            f"Type: {type(first_document)}"
        )

        print()
        print(
            "Metadata:"
        )

        print(
            first_document.metadata
        )

        print()
        print(
            "Text:"
        )

        print(
            first_document.page_content[:1000]
        )

    # --------------------------------------------------------
    # SAMPLE CLEANED PAGE
    # --------------------------------------------------------

    print()
    print(
        "=============================="
    )
    print(
        "SAMPLE CLEANED PAGE"
    )
    print(
        "=============================="
    )

    sample_found = False

    for document in documents:

        source = document.metadata.get(
            "source",
            "",
        )

        page = document.metadata.get(
            "page",
            -1,
        )

        if (
            source.endswith(
                "Deep_Learning.pdf"
            )
            and page == 38
        ):

            print(
                f"Source: {source}"
            )

            print(
                f"Page: {page}"
            )

            print(
                f"Human page: "
                f"{document.metadata.get('human_page')}"
            )

            print()
            print(
                "Text:"
            )

            print(
                document.page_content[:3000]
            )

            sample_found = True

            break

    if not sample_found:

        print(
            "Sample page was not found."
        )