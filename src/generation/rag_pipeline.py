import json
import math
import re

from sentence_transformers import CrossEncoder

from src.retrieval.dense_retriever import (
    dense_search,
)

from src.retrieval.bm25_retriever import (
    create_bm25_index,
    bm25_search,
)

from src.generation.groq_llm import (
    generate_answer,
)


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_TOP_K = 50
BM25_TOP_K = 50

HYBRID_TOP_K = 30

RERANK_TOP_K = 5

RRF_K = 60

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ------------------------------------------------------------
# Confidence configuration
# ------------------------------------------------------------

MIN_RETRIEVAL_CONFIDENCE = 0.35

HIGH_CONFIDENCE_THRESHOLD = 0.70

MEDIUM_CONFIDENCE_THRESHOLD = 0.45

# ------------------------------------------------------------
# Citation configuration
# ------------------------------------------------------------

CITATION_PATTERN = re.compile(
    r"\[(\d+)\]"
)

NO_ANSWER = (
    "I could not find the answer "
    "in the provided documents."
)


# ============================================================
# LOAD RERANKER
# ============================================================

_reranker = None


def get_reranker():
    """
    Load the cross-encoder reranker only once.
    """

    global _reranker

    if _reranker is None:

        print(
            "\nLoading reranker model..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print(
            "Reranker model loaded successfully."
        )

    return _reranker


# ============================================================
# RESULT TEXT EXTRACTION
# ============================================================

def get_result_text(result):
    """
    Extract text from a retrieval result.

    Supports:
        result["text"]
        result["document"].page_content
        result["document"]
        result["page_content"]
    """

    if result.get("text"):

        return str(
            result["text"]
        )

    document = result.get(
        "document"
    )

    if document is not None:

        if hasattr(
            document,
            "page_content",
        ):

            return str(
                document.page_content
            )

        if isinstance(
            document,
            str,
        ):

            return document

    if result.get(
        "page_content"
    ):

        return str(
            result["page_content"]
        )

    return ""


# ============================================================
# RESULT KEY
# ============================================================

def get_result_key(result):
    """
    Create a stable identifier for the same
    document chunk across dense and BM25.

    Prefer the chunk ID.

    Otherwise use:
        source + page + text
    """

    result_id = result.get(
        "id"
    )

    if result_id:

        return str(
            result_id
        )

    metadata = result.get(
        "metadata",
        {},
    )

    source = metadata.get(
        "source",
        "",
    )

    page = metadata.get(
        "page",
        "",
    )

    text = get_result_text(
        result
    )

    return (
        f"{source}|"
        f"{page}|"
        f"{text[:200]}"
    )


# ============================================================
# RRF HYBRID RETRIEVAL
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=HYBRID_TOP_K,
    k=RRF_K,
):
    """
    Combine dense and BM25 rankings using
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    Documents appearing in both retrieval
    systems receive contributions from both.
    """

    fused = {}

    # ========================================================
    # DENSE RESULTS
    # ========================================================

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        key = get_result_key(
            result
        )

        if key not in fused:

            fused[key] = {
                "id": result.get(
                    "id"
                ),
                "text": get_result_text(
                    result
                ),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_score": result.get(
                    "score"
                ),
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # ========================================================
    # BM25 RESULTS
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        key = get_result_key(
            result
        )

        if key not in fused:

            fused[key] = {
                "id": result.get(
                    "id"
                ),
                "text": get_result_text(
                    result
                ),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_score": None,
                "bm25_score": result.get(
                    "score"
                ),
                "rrf_score": 0.0,
            }

        else:

            fused[key]["bm25_rank"] = rank

            fused[key]["bm25_score"] = (
                result.get(
                    "score"
                )
            )

            if not fused[key].get(
                "text"
            ):

                fused[key]["text"] = (
                    get_result_text(
                        result
                    )
                )

            if not fused[key].get(
                "document"
            ):

                fused[key]["document"] = (
                    result.get(
                        "document"
                    )
                )

            if not fused[key].get(
                "metadata"
            ):

                fused[key]["metadata"] = (
                    result.get(
                        "metadata",
                        {},
                    )
                )

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # ========================================================
    # SORT
    # ========================================================

    results = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return results[
        :top_k
    ]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def create_hybrid_results(
    query,
    bm25,
    chunks,
    top_k=HYBRID_TOP_K,
):
    """
    Dense
        +
    BM25
        ↓
    RRF
    """

    print(
        "\n=============================="
    )

    print(
        "RUNNING DENSE RETRIEVAL"
    )

    print(
        "=============================="
    )

    dense_results = dense_search(
        query,
        top_k=DENSE_TOP_K,
    )

    print(
        f"Dense results: "
        f"{len(dense_results)}"
    )

    print(
        "\n=============================="
    )

    print(
        "RUNNING BM25 RETRIEVAL"
    )

    print(
        "=============================="
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=BM25_TOP_K,
    )

    print(
        f"BM25 results: "
        f"{len(bm25_results)}"
    )

    print(
        "\n=============================="
    )

    print(
        "COMBINING WITH RRF"
    )

    print(
        "=============================="
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k,
        k=RRF_K,
    )

    print(
        f"Hybrid results: "
        f"{len(hybrid_results)}"
    )

    return hybrid_results


# ============================================================
# RERANK
# ============================================================

def rerank_results(
    query,
    results,
    top_k=RERANK_TOP_K,
):
    """
    Rerank hybrid candidates using:

        cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    if not results:

        return []

    model = get_reranker()

    print(
        f"\nReranking "
        f"{len(results)} candidates..."
    )

    pairs = []

    valid_results = []

    for result in results:

        text = get_result_text(
            result
        )

        if not text.strip():

            continue

        pairs.append(
            [
                query,
                text,
            ]
        )

        valid_results.append(
            result
        )

    if not pairs:

        return []

    scores = model.predict(
        pairs
    )

    reranked = []

    for result, score in zip(
        valid_results,
        scores,
    ):

        item = dict(
            result
        )

        item[
            "reranker_score"
        ] = float(score)

        reranked.append(
            item
        )

    reranked.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked[
        :top_k
    ]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):
    """
    Build numbered context blocks.

    The number becomes the citation ID:

        Context 1 -> [1]
        Context 2 -> [2]
        Context 3 -> [3]

    This gives the LLM an explicit citation map.
    """

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        human_page = metadata.get(
            "human_page"
        )

        text = get_result_text(
            result
        )

        if not text.strip():

            continue

        if human_page is not None:

            page_info = (
                f"Page: {page} "
                f"(Human page: {human_page})"
            )

        else:

            page_info = (
                f"Page: {page}"
            )

        reranker_score = result.get(
            "reranker_score"
        )

        if reranker_score is not None:

            score_info = (
                f"\nReranker score: "
                f"{reranker_score:.4f}"
            )

        else:

            score_info = ""

        context_parts.append(
            f"""
--- Context {i} ---

Citation ID: [{i}]

Source: {source}

{page_info}{score_info}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# RAG PROMPT
# ============================================================

def create_rag_prompt(
    question,
    context,
):
    """
    Grounded generation prompt.

    Forces:
        - document-only answers
        - explicit citations
        - no invented citations
        - explicit unknown handling
    """

    return f"""
You are a document question-answering assistant.

Your job is to answer the user's question using
ONLY the supplied document context.

IMPORTANT RULES:

1. Use ONLY information contained in the supplied
   document context.

2. Do NOT use outside knowledge.

3. Do NOT invent facts.

4. Do NOT make assumptions that are not supported
   by the documents.

5. Every factual claim in your answer MUST have
   at least one citation.

6. Citations MUST use the exact format:

   [1]
   [2]
   [3]

7. A citation number refers ONLY to the matching
   Context number.

8. NEVER create a citation number that does not
   exist in the supplied context.

9. If multiple context blocks support a claim,
   you may use multiple citations, for example:

   [1][3]

10. Prefer the context that directly answers the
    question.

11. Keep the answer clear and concise.

12. If several context sections support the answer,
    combine them carefully.

13. If the answer is not present in the supplied
    context, return exactly:

I could not find the answer in the provided documents.

14. Do not cite a context block merely because it
    is related. Cite it only when it supports the
    factual claim.

15. Do not include a separate bibliography.
    Inline citations are required.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_citations(
    answer
):
    """
    Extract citation numbers from generated answer.

    Example:

        "Machine learning learns from data [1]."

    returns:

        [1]
    """

    if not answer:

        return []

    citations = []

    for match in CITATION_PATTERN.finditer(
        answer
    ):

        number = int(
            match.group(1)
        )

        if number not in citations:

            citations.append(
                number
            )

    return citations


# ============================================================
# CITATION CLAIM EXTRACTION
# ============================================================

def extract_citation_claims(
    answer
):
    """
    Associate each citation with the sentence
    containing that citation.

    Example:

        "Machine learning learns from data [1]."

    becomes approximately:

        {
            1: "Machine learning learns from data [1]."
        }
    """

    claims = {}

    if not answer:

        return claims

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    for sentence in sentences:

        citations = extract_citations(
            sentence
        )

        for citation in citations:

            claims.setdefault(
                citation,
                [],
            ).append(
                sentence.strip()
            )

    return claims


# ============================================================
# CITATION METADATA
# ============================================================

def build_citation_map(
    results
):
    """
    Build:

        [1] -> source/page/chunk
        [2] -> source/page/chunk
        ...

    This is used by the API/dashboard later.
    """

    citation_map = {}

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        citation_map[index] = {
            "citation": f"[{index}]",
            "id": result.get(
                "id"
            ),
            "source": metadata.get(
                "source",
                "Unknown",
            ),
            "page": metadata.get(
                "page",
                "Unknown",
            ),
            "human_page": metadata.get(
                "human_page"
            ),
            "text": get_result_text(
                result
            ),
            "reranker_score": result.get(
                "reranker_score"
            ),
            "rrf_score": result.get(
                "rrf_score"
            ),
            "dense_rank": result.get(
                "dense_rank"
            ),
            "bm25_rank": result.get(
                "bm25_rank"
            ),
        }

    return citation_map


# ============================================================
# LLM CITATION VERIFICATION
# ============================================================

def verify_citations_with_llm(
    answer,
    citation_map,
):
    """
    Verify generated citation/claim pairs.

    The verifier receives:
        - generated answer
        - cited source chunk
        - claim containing citation

    It returns a structured verification result.

    If the LLM judge cannot be parsed, we fall back
    to deterministic text-overlap verification.
    """

    citation_claims = (
        extract_citation_claims(
            answer
        )
    )

    if not citation_claims:

        return {
            "verified": [],
            "unsupported": [],
            "invalid": [],
        }

    verified = []

    unsupported = []

    invalid = []

    for citation, claims in (
        citation_claims.items()
    ):

        if citation not in citation_map:

            invalid.append(
                citation
            )

            continue

        source = citation_map[
            citation
        ]

        source_text = source.get(
            "text",
            "",
        )

        for claim in claims:

            prompt = f"""
You are a citation verification judge.

Determine whether the provided source chunk
actually supports the claim.

SOURCE CHUNK:

{source_text}

CLAIM:

{claim}

Return ONLY valid JSON in this exact format:

{{
  "supported": true,
  "reason": "short explanation"
}}

or:

{{
  "supported": false,
  "reason": "short explanation"
}}

Do not use outside knowledge.
Judge only whether the source supports the claim.
"""

            try:

                raw = generate_answer(
                    prompt
                )

                parsed = _parse_json_object(
                    raw
                )

                supported = bool(
                    parsed.get(
                        "supported",
                        False,
                    )
                )

            except Exception:

                supported = (
                    _deterministic_claim_support(
                        claim,
                        source_text,
                    )
                )

            item = {
                "citation": citation,
                "claim": claim,
                "source": source.get(
                    "source",
                    "Unknown",
                ),
                "page": source.get(
                    "page",
                    "Unknown",
                ),
                "supported": supported,
            }

            if supported:

                verified.append(
                    item
                )

            else:

                unsupported.append(
                    item
                )

    return {
        "verified": verified,
        "unsupported": unsupported,
        "invalid": invalid,
    }


# ============================================================
# JSON PARSER
# ============================================================

def _parse_json_object(
    text
):
    """
    Parse JSON from an LLM response.

    Handles:
        pure JSON
        ```json ... ```
        extra text around JSON
    """

    if not text:

        raise ValueError(
            "Empty LLM response."
        )

    cleaned = text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:

        value = json.loads(
            cleaned
        )

        if not isinstance(
            value,
            dict,
        ):

            raise ValueError(
                "JSON response is not an object."
            )

        return value

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:

            raise

        value = json.loads(
            match.group(0)
        )

        if not isinstance(
            value,
            dict,
        ):

            raise ValueError(
                "JSON response is not an object."
            )

        return value


# ============================================================
# DETERMINISTIC CITATION FALLBACK
# ============================================================

def _normalize_words(
    text
):
    """
    Normalize text into a set of useful words.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]{3,}\b",
        text.lower(),
    )

    stop_words = {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "into",
        "are",
        "was",
        "were",
        "for",
        "have",
        "has",
        "had",
        "their",
        "there",
        "which",
        "about",
        "using",
        "used",
        "can",
        "may",
        "will",
        "does",
        "did",
    }

    return {
        word
        for word in words
        if word not in stop_words
    }


def _deterministic_claim_support(
    claim,
    source_text,
):
    """
    Conservative fallback when the LLM verifier
    cannot be used.

    It checks meaningful word overlap.

    This is NOT treated as equivalent to the
    LLM verification layer.
    """

    claim_words = _normalize_words(
        claim
    )

    source_words = _normalize_words(
        source_text
    )

    if not claim_words:

        return False

    overlap = (
        claim_words
        & source_words
    )

    ratio = (
        len(overlap)
        / len(claim_words)
    )

    return ratio >= 0.35


# ============================================================
# RETRIEVAL CONFIDENCE
# ============================================================

def sigmoid(
    value
):
    """
    Numerically stable sigmoid.
    """

    try:

        if value >= 0:

            z = math.exp(
                -value
            )

            return 1.0 / (
                1.0 + z
            )

        z = math.exp(
            value
        )

        return z / (
            1.0 + z
        )

    except OverflowError:

        return (
            0.0
            if value < 0
            else 1.0
        )


def calculate_retrieval_confidence(
    results
):
    """
    Estimate retrieval confidence from reranker
    scores.

    The CrossEncoder score itself is not treated as
    a probability, so we normalize it with sigmoid.

    More weight is given to the highest-ranked chunks.
    """

    if not results:

        return 0.0

    scores = []

    for result in results:

        score = result.get(
            "reranker_score"
        )

        if score is None:

            continue

        scores.append(
            float(score)
        )

    if not scores:

        return 0.0

    normalized = [
        sigmoid(score)
        for score in scores
    ]

    # --------------------------------------------------------
    # Weighted average
    # --------------------------------------------------------

    weights = [
        1.0 / (index + 1)
        for index in range(
            len(normalized)
        )
    ]

    weighted_sum = sum(
        score * weight
        for score, weight in zip(
            normalized,
            weights,
        )
    )

    weight_sum = sum(
        weights
    )

    confidence = (
        weighted_sum
        / weight_sum
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        ),
        4,
    )


# ============================================================
# CITATION COVERAGE
# ============================================================

def calculate_citation_coverage(
    answer,
    verification,
):
    """
    Calculate citation coverage.

    Coverage is based on cited claims that were
    actually verified.

    Invalid and unsupported citations do not count.
    """

    claims = (
        extract_citation_claims(
            answer
        )
    )

    total_claims = sum(
        len(value)
        for value in claims.values()
    )

    if total_claims == 0:

        return 0.0

    verified_count = len(
        verification.get(
            "verified",
            [],
        )
    )

    coverage = (
        verified_count
        / total_claims
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                coverage,
            ),
        ),
        4,
    )


# ============================================================
# ANSWER COMPLETENESS
# ============================================================

def calculate_answer_completeness(
    question,
    answer,
):
    """
    Estimate whether the answer addresses the
    important terms in the question.

    This is a lightweight deterministic signal.
    It does not claim to understand the question
    semantically.
    """

    question_words = _normalize_words(
        question
    )

    answer_words = _normalize_words(
        answer
    )

    if not question_words:

        return 0.0

    overlap = (
        question_words
        & answer_words
    )

    ratio = (
        len(overlap)
        / len(question_words)
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                ratio,
            ),
        ),
        4,
    )


# ============================================================
# COMPOSITE CONFIDENCE
# ============================================================

def calculate_confidence_score(
    retrieval_confidence,
    citation_coverage,
    answer_completeness,
):
    """
    Composite answer confidence.

    Weighting:

        Retrieval confidence: 40%
        Citation coverage:    40%
        Completeness:          20%
    """

    score = (
        0.40
        * retrieval_confidence
        +
        0.40
        * citation_coverage
        +
        0.20
        * answer_completeness
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        ),
        4,
    )


# ============================================================
# CONFIDENCE LABEL
# ============================================================

def confidence_label(
    score
):
    """
    Convert confidence score into a human-readable
    label.
    """

    if score >= HIGH_CONFIDENCE_THRESHOLD:

        return "high"

    if score >= MEDIUM_CONFIDENCE_THRESHOLD:

        return "medium"

    return "low"


# ============================================================
# SOURCE BUILDING
# ============================================================

def build_sources(
    citation_map,
    verification,
):
    """
    Build clean source metadata for the API/dashboard.

    Only verified citations are marked as verified.
    """

    verified_numbers = {
        item["citation"]
        for item in verification.get(
            "verified",
            [],
        )
    }

    unsupported_numbers = {
        item["citation"]
        for item in verification.get(
            "unsupported",
            [],
        )
    }

    sources = []

    for citation, source in (
        citation_map.items()
    ):

        sources.append(
            {
                "citation": (
                    f"[{citation}]"
                ),
                "source": source.get(
                    "source",
                    "Unknown",
                ),
                "page": source.get(
                    "page",
                    "Unknown",
                ),
                "human_page": source.get(
                    "human_page"
                ),
                "id": source.get(
                    "id"
                ),
                "reranker_score": source.get(
                    "reranker_score"
                ),
                "rrf_score": source.get(
                    "rrf_score"
                ),
                "verified": (
                    citation
                    in verified_numbers
                ),
                "unsupported": (
                    citation
                    in unsupported_numbers
                ),
            }
        )

    return sources


# ============================================================
# HANDLE UNSUPPORTED CITATIONS
# ============================================================

def remove_invalid_citations(
    answer,
    invalid_citations,
):
    """
    Remove citations that reference a context block
    that does not exist.

    Example:

        [99]

    is removed if Context 99 does not exist.
    """

    if not invalid_citations:

        return answer

    invalid_set = {
        int(value)
        for value in invalid_citations
    }

    def replacement(
        match
    ):

        number = int(
            match.group(1)
        )

        if number in invalid_set:

            return ""

        return match.group(0)

    return CITATION_PATTERN.sub(
        replacement,
        answer,
    )


# ============================================================
# FORMAT VERIFICATION WARNING
# ============================================================

def append_verification_warning(
    answer,
    verification,
):
    """
    Append a transparent warning when generated
    citations are unsupported.

    We do not silently pretend unsupported citations
    are valid.
    """

    unsupported = verification.get(
        "unsupported",
        [],
    )

    invalid = verification.get(
        "invalid",
        [],
    )

    if not unsupported and not invalid:

        return answer

    lines = [
        answer.rstrip(),
        "",
        "Citation verification warning:",
    ]

    if unsupported:

        numbers = sorted(
            {
                item["citation"]
                for item in unsupported
            }
        )

        lines.append(
            "Unsupported citations: "
            + ", ".join(
                f"[{number}]"
                for number in numbers
            )
        )

    if invalid:

        lines.append(
            "Invalid citations: "
            + ", ".join(
                f"[{number}]"
                for number in sorted(
                    invalid
                )
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# RESULT DETAILS
# ============================================================

def build_result_details(
    question,
    answer,
    reranked_results,
    verification,
):
    """
    Build the structured result used later by:

        FastAPI
        Streamlit
        evaluation
        logging
    """

    citation_map = build_citation_map(
        reranked_results
    )

    retrieval_confidence = (
        calculate_retrieval_confidence(
            reranked_results
        )
    )

    citation_coverage = (
        calculate_citation_coverage(
            answer,
            verification,
        )
    )

    answer_completeness = (
        calculate_answer_completeness(
            question,
            answer,
        )
    )

    composite_confidence = (
        calculate_confidence_score(
            retrieval_confidence,
            citation_coverage,
            answer_completeness,
        )
    )

    return {
        "question": question,
        "answer": answer,
        "confidence": composite_confidence,
        "confidence_label": confidence_label(
            composite_confidence
        ),
        "retrieval_confidence": (
            retrieval_confidence
        ),
        "citation_coverage": (
            citation_coverage
        ),
        "answer_completeness": (
            answer_completeness
        ),
        "citations": extract_citations(
            answer
        ),
        "sources": build_sources(
            citation_map,
            verification,
        ),
        "citation_verification": verification,
        "retrieved_chunks": len(
            reranked_results
        ),
    }


# ============================================================
# COMPLETE RAG ANSWER
# ============================================================

def rag_answer(
    question,
    bm25,
    chunks,
    return_details=False,
):
    """
    Complete production-oriented RAG pipeline:

        Question
            ↓
        Dense Retrieval
            ↓
        BM25
            ↓
        RRF
            ↓
        Cross-Encoder
            ↓
        Top 5
            ↓
        Numbered Context
            ↓
        Grounded Groq Generation
            ↓
        Citation Extraction
            ↓
        Citation Verification
            ↓
        Confidence Scoring
            ↓
        Answer + Sources

    Backward compatibility:

        return_details=False

    returns only the answer string.

    When:

        return_details=True

    returns a structured dictionary containing:

        answer
        confidence
        confidence_label
        retrieval_confidence
        citation_coverage
        answer_completeness
        citations
        sources
        citation_verification
        retrieved_chunks
    """

    # ========================================================
    # QUESTION VALIDATION
    # ========================================================

    question = str(
        question
    ).strip()

    if not question:

        answer = NO_ANSWER

        if return_details:

            return {
                "question": question,
                "answer": answer,
                "confidence": 0.0,
                "confidence_label": "low",
                "retrieval_confidence": 0.0,
                "citation_coverage": 0.0,
                "answer_completeness": 0.0,
                "citations": [],
                "sources": [],
                "citation_verification": {
                    "verified": [],
                    "unsupported": [],
                    "invalid": [],
                },
                "retrieved_chunks": 0,
            }

        return answer

    # ========================================================
    # STEP 1: HYBRID RETRIEVAL
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 1: HYBRID RETRIEVAL"
    )

    print(
        "=============================="
    )

    hybrid_results = create_hybrid_results(
        query=question,
        bm25=bm25,
        chunks=chunks,
        top_k=HYBRID_TOP_K,
    )

    if not hybrid_results:

        answer = NO_ANSWER

        if return_details:

            return build_result_details(
                question=question,
                answer=answer,
                reranked_results=[],
                verification={
                    "verified": [],
                    "unsupported": [],
                    "invalid": [],
                },
            )

        return answer

    # ========================================================
    # STEP 2: RERANK
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 2: CROSS-ENCODER RERANKING"
    )

    print(
        "=============================="
    )

    reranked_results = rerank_results(
        question,
        hybrid_results,
        top_k=RERANK_TOP_K,
    )

    print(
        f"Reranked results: "
        f"{len(reranked_results)}"
    )

    if not reranked_results:

        answer = NO_ANSWER

        if return_details:

            return build_result_details(
                question=question,
                answer=answer,
                reranked_results=[],
                verification={
                    "verified": [],
                    "unsupported": [],
                    "invalid": [],
                },
            )

        return answer

    # ========================================================
    # STEP 3: RETRIEVAL CONFIDENCE
    # ========================================================

    retrieval_confidence = (
        calculate_retrieval_confidence(
            reranked_results
        )
    )

    print(
        "\n=============================="
    )

    print(
        "RETRIEVAL CONFIDENCE"
    )

    print(
        "=============================="
    )

    print(
        f"Retrieval confidence: "
        f"{retrieval_confidence:.4f}"
    )

    # ========================================================
    # STEP 4: LOW-CONFIDENCE GUARD
    # ========================================================

    if (
        retrieval_confidence
        < MIN_RETRIEVAL_CONFIDENCE
    ):

        print(
            "\nLow retrieval confidence."
        )

        print(
            "Skipping generation."
        )

        answer = (
            "I could not find the answer "
            "in the provided documents."
        )

        if return_details:

            return build_result_details(
                question=question,
                answer=answer,
                reranked_results=reranked_results,
                verification={
                    "verified": [],
                    "unsupported": [],
                    "invalid": [],
                },
            )

        return answer

    # ========================================================
    # STEP 5: DISPLAY RETRIEVED SOURCES
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "RERANKED RESULTS"
    )

    print(
        "=============================="
    )

    for i, result in enumerate(
        reranked_results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        print(
            f"\n--- Context {i} ---"
        )

        print(
            "ID:",
            result.get(
                "id"
            ),
        )

        print(
            "Reranker score:",
            result.get(
                "reranker_score"
            ),
        )

        print(
            "RRF score:",
            result.get(
                "rrf_score"
            ),
        )

        print(
            "Source:",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        print(
            "Page:",
            metadata.get(
                "page",
                "Unknown",
            ),
        )

    # ========================================================
    # STEP 6: BUILD CONTEXT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 3: BUILDING NUMBERED CONTEXT"
    )

    print(
        "=============================="
    )

    context = build_context(
        reranked_results
    )

    if not context.strip():

        answer = NO_ANSWER

        if return_details:

            return build_result_details(
                question=question,
                answer=answer,
                reranked_results=reranked_results,
                verification={
                    "verified": [],
                    "unsupported": [],
                    "invalid": [],
                },
            )

        return answer

    print(
        context
    )

    # ========================================================
    # STEP 7: CREATE GROUNDED PROMPT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 4: CREATING GROUNDED PROMPT"
    )

    print(
        "=============================="
    )

    prompt = create_rag_prompt(
        question=question,
        context=context,
    )

    # ========================================================
    # STEP 8: GROQ GENERATION
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 5: GROUNDED GROQ GENERATION"
    )

    print(
        "=============================="
    )

    answer = generate_answer(
        prompt
    )

    if not answer:

        answer = NO_ANSWER

    answer = str(
        answer
    ).strip()

    # ========================================================
    # STEP 9: EXTRACT CITATIONS
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 6: EXTRACTING CITATIONS"
    )

    print(
        "=============================="
    )

    citations = extract_citations(
        answer
    )

    print(
        "Generated citations:",
        citations,
    )

    citation_map = build_citation_map(
        reranked_results
    )

    # ========================================================
    # STEP 10: VERIFY CITATIONS
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "STEP 7: VERIFYING CITATIONS"
    )

    print(
        "=============================="
    )

    verification = (
        verify_citations_with_llm(
            answer=answer,
            citation_map=citation_map,
        )
    )

    print(
        "Verified citations:",
        len(
            verification.get(
                "verified",
                [],
            )
        ),
    )

    print(
        "Unsupported citations:",
        len(
            verification.get(
                "unsupported",
                [],
            )
        ),
    )

    print(
        "Invalid citations:",
        len(
            verification.get(
                "invalid",
                [],
            )
        ),
    )

    # ========================================================
    # STEP 11: REMOVE INVALID CITATIONS
    # ========================================================

    answer = remove_invalid_citations(
        answer,
        verification.get(
            "invalid",
            [],
        ),
    )

    # ========================================================
    # STEP 12: CITATION COVERAGE
    # ========================================================

    citation_coverage = (
        calculate_citation_coverage(
            answer,
            verification,
        )
    )

    print(
        "\nCitation coverage:",
        f"{citation_coverage:.4f}",
    )

    # ========================================================
    # STEP 13: ANSWER COMPLETENESS
    # ========================================================

    answer_completeness = (
        calculate_answer_completeness(
            question,
            answer,
        )
    )

    print(
        "Answer completeness:",
        f"{answer_completeness:.4f}",
    )

    # ========================================================
    # STEP 14: COMPOSITE CONFIDENCE
    # ========================================================

    composite_confidence = (
        calculate_confidence_score(
            retrieval_confidence,
            citation_coverage,
            answer_completeness,
        )
    )

    label = confidence_label(
        composite_confidence
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL CONFIDENCE"
    )

    print(
        "=============================="
    )

    print(
        f"Retrieval confidence: "
        f"{retrieval_confidence:.4f}"
    )

    print(
        f"Citation coverage: "
        f"{citation_coverage:.4f}"
    )

    print(
        f"Answer completeness: "
        f"{answer_completeness:.4f}"
    )

    print(
        f"Composite confidence: "
        f"{composite_confidence:.4f}"
    )

    print(
        f"Confidence label: "
        f"{label}"
    )

    # ========================================================
    # STEP 15: VERIFICATION WARNING
    # ========================================================

    answer = append_verification_warning(
        answer,
        verification,
    )

    # ========================================================
    # STEP 16: STRUCTURED RESULT
    # ========================================================

    details = build_result_details(
        question=question,
        answer=answer,
        reranked_results=reranked_results,
        verification=verification,
    )

    if return_details:

        return details

    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    return answer


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "             RAG PIPELINE"
    )

    print(
        "========================================"
    )

    print(
        "\nCreating BM25 index..."
    )

    bm25, chunks = create_bm25_index()

    print(
        "\nBM25 index ready."
    )

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    while True:

        question = input(
            "\nEnter your question "
            "(or 'exit' to quit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nExiting RAG pipeline."
            )

            break

        if not question:

            continue

        try:

            result = rag_answer(
                question=question,
                bm25=bm25,
                chunks=chunks,
                return_details=True,
            )

        except Exception as error:

            print(
                "\n=============================="
            )

            print(
                "ERROR"
            )

            print(
                "=============================="
            )

            print(
                type(error).__name__
            )

            print(
                error
            )

            continue

        print(
            "\n========================================"
        )

        print(
            "             FINAL RAG ANSWER"
        )

        print(
            "========================================"
        )

        print(
            result["answer"]
        )

        print(
            "\n----------------------------------------"
        )

        print(
            "Confidence:",
            result["confidence"],
        )

        print(
            "Confidence label:",
            result["confidence_label"],
        )

        print(
            "Retrieval confidence:",
            result["retrieval_confidence"],
        )

        print(
            "Citation coverage:",
            result["citation_coverage"],
        )

        print(
            "Answer completeness:",
            result["answer_completeness"],
        )

        print(
            "\nSources:"
        )

        for source in result[
            "sources"
        ]:

            print(
                f"{source['citation']} "
                f"{source['source']} "
                f"(page {source['page']}) "
                f"verified={source['verified']}"
            )

        print(
            "\n========================================"
        )