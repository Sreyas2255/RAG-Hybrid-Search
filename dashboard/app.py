import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Streamlit will run inside Docker.
# "rag-api" is the Docker container name and is resolvable
# through the shared Docker network: rag-network.
API_URL = "http://rag-api:8000"

ASK_ENDPOINT = f"{API_URL}/v1/ask"
HEALTH_ENDPOINT = f"{API_URL}/health"
STATUS_ENDPOINT = f"{API_URL}/v1/status"
DOCUMENTS_ENDPOINT = f"{API_URL}/v1/documents"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(obj, key, default=None):
    """
    Safely get a value from either:
        - dictionary
        - object
        - None
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def normalize_citation_id(citation):
    """
    Convert different citation formats into an integer ID.

    Supported examples:

        1
        "1"
        "[1]"
        {"citation_id": 1}
        {"id": 1}
        {"citation": "[1]"}
    """

    # Integer
    if isinstance(citation, int):
        return citation

    # String
    if isinstance(citation, str):

        value = citation.strip()

        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].strip()

        try:
            return int(value)
        except ValueError:
            return None

    # Dictionary / object
    citation_id = safe_get(
        citation,
        "citation_id",
    )

    if citation_id is None:
        citation_id = safe_get(
            citation,
            "id",
        )

    if citation_id is None:
        citation_id = safe_get(
            citation,
            "citation",
        )

    return normalize_citation_id(citation_id)


def normalize_citations(citations):
    """
    Normalize API citations.

    Example:

        [1, 5, 2]

    Returns:

        [1, 5, 2]
    """

    if not isinstance(citations, list):
        return []

    normalized = []

    for citation in citations:

        citation_id = normalize_citation_id(
            citation
        )

        if citation_id is not None:

            if citation_id not in normalized:
                normalized.append(citation_id)

    return normalized


def normalize_sources(sources):
    """
    Normalize source metadata.

    Returns a list of dictionaries.
    """

    if not isinstance(sources, list):
        return []

    normalized = []

    for source in sources:

        if isinstance(source, dict):

            normalized.append(
                {
                    "citation": source.get(
                        "citation"
                    ),
                    "source": source.get(
                        "source",
                        "Unknown",
                    ),
                    "page": source.get(
                        "page"
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
                    "verified": source.get(
                        "verified",
                        False,
                    ),
                    "unsupported": source.get(
                        "unsupported",
                        False,
                    ),
                }
            )

        else:

            normalized.append(
                {
                    "citation": None,
                    "source": str(source),
                    "page": None,
                    "human_page": None,
                    "id": None,
                    "reranker_score": None,
                    "rrf_score": None,
                    "verified": False,
                    "unsupported": False,
                }
            )

    return normalized


def citation_number_from_source(source):
    """
    Extract citation number from source metadata.

    Examples:

        "[1]" -> 1
        1     -> 1
        "1"   -> 1
    """

    citation = source.get("citation")

    return normalize_citation_id(citation)


def build_source_map(sources):
    """
    Build:

        citation number -> source metadata
    """

    source_map = {}

    for source in sources:

        citation_id = citation_number_from_source(
            source
        )

        if citation_id is not None:

            source_map[citation_id] = source

    return source_map


def confidence_label(confidence):
    """
    Convert confidence score into a label.
    """

    if confidence is None:
        return "N/A"

    try:
        score = float(confidence)
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"

    if score >= 0.70:
        return "High"

    if score >= 0.45:
        return "Medium"

    return "Low"


def confidence_percent(confidence):
    """
    Convert 0-1 confidence to percentage.
    """

    if confidence is None:
        return "N/A"

    try:
        value = float(confidence)
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"

    return f"{value * 100:.1f}%"


def get_error_detail(response):
    """
    Safely extract FastAPI error detail.
    """

    try:

        data = response.json()

        if isinstance(data, dict):

            return data.get(
                "detail",
                "Unknown API error.",
            )

        return str(data)

    except ValueError:

        if response.text:
            return response.text

        return "Unknown API error."


# ============================================================
# TITLE
# ============================================================

st.title(
    "📚 Hybrid RAG Assistant"
)

st.write(
    "Ask questions about the documents indexed by the "
    "Hybrid Retrieval-Augmented Generation system."
)


# ============================================================
# API STATUS
# ============================================================

api_online = False
status_data = {}
health_data = {}
documents_data = {}


try:

    health_response = requests.get(
        HEALTH_ENDPOINT,
        timeout=5,
    )

    if health_response.status_code == 200:

        health_data = health_response.json()

        if health_data.get(
            "bm25_ready",
            False,
        ):

            api_online = True

            st.success(
                "🟢 RAG API is online and ready."
            )

        else:

            st.warning(
                "🟡 RAG API is running but "
                "the RAG system is not ready."
            )

    else:

        st.warning(
            "RAG API returned an unexpected "
            "health response."
        )

except requests.RequestException as error:

    st.error(
        "🔴 Could not connect to the FastAPI server."
    )

    st.caption(
        f"Trying to connect to: {API_URL}"
    )

    st.caption(
        f"Connection error: {error}"
    )


# ============================================================
# GET STATUS
# ============================================================

if api_online:

    try:

        status_response = requests.get(
            STATUS_ENDPOINT,
            timeout=5,
        )

        if status_response.status_code == 200:

            status_data = status_response.json()

    except requests.RequestException:

        status_data = {}


# ============================================================
# QUESTION SECTION
# ============================================================

st.subheader(
    "Ask a question"
)

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What is deep learning?"
    ),
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "🔎 Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not api_online:

        st.error(
            "The RAG API is not ready."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                response = requests.post(
                    ASK_ENDPOINT,
                    json={
                        "question": question.strip()
                    },
                    timeout=180,
                )

                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    data = response.json()

                    # -------------------------------------------------
                    # Basic fields
                    # -------------------------------------------------

                    answer = data.get(
                        "answer",
                        "No answer returned.",
                    )

                    returned_question = data.get(
                        "question",
                        question.strip(),
                    )

                    confidence = data.get(
                        "confidence"
                    )

                    citations = normalize_citations(
                        data.get(
                            "citations",
                            [],
                        )
                    )

                    sources = normalize_sources(
                        data.get(
                            "sources",
                            [],
                        )
                    )

                    citation_verification = data.get(
                        "citation_verification",
                        {},
                    )

                    retrieved_chunks = data.get(
                        "retrieved_chunks",
                        0,
                    )

                    # -------------------------------------------------
                    # Source map
                    # -------------------------------------------------

                    source_map = build_source_map(
                        sources
                    )

                    # =================================================
                    # ANSWER
                    # =================================================

                    st.divider()

                    st.subheader(
                        "💡 Answer"
                    )

                    st.write(
                        answer
                    )

                    # =================================================
                    # ANSWER QUALITY
                    # =================================================

                    st.subheader(
                        "📊 Answer Quality"
                    )

                    col1, col2, col3 = st.columns(3)

                    # -------------------------------------------------
                    # Confidence
                    # -------------------------------------------------

                    with col1:

                        st.metric(
                            "Confidence",
                            confidence_percent(
                                confidence
                            ),
                        )

                        label = confidence_label(
                            confidence
                        )

                        if label == "High":

                            st.success(
                                f"⬆ {label}"
                            )

                        elif label == "Medium":

                            st.warning(
                                f"→ {label}"
                            )

                        elif label == "Low":

                            st.error(
                                f"⬇ {label}"
                            )

                        else:

                            st.info(
                                "N/A"
                            )

                    # -------------------------------------------------
                    # Citations
                    # -------------------------------------------------

                    with col2:

                        st.metric(
                            "Citations",
                            len(citations),
                        )

                    # -------------------------------------------------
                    # Sources
                    # -------------------------------------------------

                    with col3:

                        st.metric(
                            "Sources",
                            len(sources),
                        )

                    # =================================================
                    # CITATIONS
                    # =================================================

                    st.subheader(
                        "📚 Citations"
                    )

                    if not citations:

                        st.info(
                            "No structured citation metadata "
                            "was returned by the API."
                        )

                    else:

                        for citation_id in citations:

                            source = source_map.get(
                                citation_id
                            )

                            with st.container():

                                st.markdown(
                                    f"### Citation [{citation_id}]"
                                )

                                # -----------------------------------------
                                # Source metadata
                                # -----------------------------------------

                                if source:

                                    source_name = source.get(
                                        "source",
                                        "Unknown",
                                    )

                                    page = source.get(
                                        "page"
                                    )

                                    human_page = source.get(
                                        "human_page"
                                    )

                                    verified = source.get(
                                        "verified",
                                        False,
                                    )

                                    unsupported = source.get(
                                        "unsupported",
                                        False,
                                    )

                                    st.markdown(
                                        f"**Source:** "
                                        f"`{source_name}`"
                                    )

                                    if page is not None:

                                        page_text = (
                                            f"Page {page}"
                                        )

                                        if human_page is not None:

                                            page_text += (
                                                f" "
                                                f"(Human page "
                                                f"{human_page})"
                                            )

                                        st.markdown(
                                            f"**Page:** "
                                            f"{page_text}"
                                        )

                                    # -------------------------------------
                                    # Verification status
                                    # -------------------------------------

                                    if verified:

                                        st.success(
                                            "✓ Citation verified"
                                        )

                                    elif unsupported:

                                        st.error(
                                            "✗ Citation unsupported"
                                        )

                                    else:

                                        st.info(
                                            "Citation metadata available"
                                        )

                                    # -------------------------------------
                                    # Scores
                                    # -------------------------------------

                                    reranker_score = source.get(
                                        "reranker_score"
                                    )

                                    rrf_score = source.get(
                                        "rrf_score"
                                    )

                                    if (
                                        reranker_score is not None
                                        or rrf_score is not None
                                    ):

                                        score_columns = st.columns(2)

                                        with score_columns[0]:

                                            if (
                                                reranker_score
                                                is not None
                                            ):

                                                st.caption(
                                                    "Reranker score"
                                                )

                                                st.write(
                                                    f"{float(reranker_score):.4f}"
                                                )

                                        with score_columns[1]:

                                            if (
                                                rrf_score
                                                is not None
                                            ):

                                                st.caption(
                                                    "RRF score"
                                                )

                                                st.write(
                                                    f"{float(rrf_score):.6f}"
                                                )

                                # -----------------------------------------
                                # No source metadata
                                # -----------------------------------------

                                else:

                                    st.info(
                                        f"Citation [{citation_id}] "
                                        "was returned, but no matching "
                                        "source metadata was returned."
                                    )

                                st.divider()

                    # =================================================
                    # RETRIEVED SOURCES
                    # =================================================

                    st.subheader(
                        "📄 Retrieved Sources"
                    )

                    if not sources:

                        st.info(
                            "No retrieved source metadata "
                            "was returned by the API."
                        )

                    else:

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            citation = source.get(
                                "citation"
                            )

                            source_name = source.get(
                                "source",
                                "Unknown",
                            )

                            page = source.get(
                                "page"
                            )

                            human_page = source.get(
                                "human_page"
                            )

                            verified = source.get(
                                "verified",
                                False,
                            )

                            unsupported = source.get(
                                "unsupported",
                                False,
                            )

                            # -----------------------------------------
                            # Title
                            # -----------------------------------------

                            if citation:

                                title = (
                                    f"{citation} "
                                    f"{source_name}"
                                )

                            else:

                                title = (
                                    f"Source {index}: "
                                    f"{source_name}"
                                )

                            with st.expander(
                                title,
                                expanded=False,
                            ):

                                st.markdown(
                                    f"**Source:** "
                                    f"`{source_name}`"
                                )

                                if page is not None:

                                    st.markdown(
                                        f"**Page:** "
                                        f"{page}"
                                    )

                                if human_page is not None:

                                    st.markdown(
                                        f"**Human page:** "
                                        f"{human_page}"
                                    )

                                chunk_id = source.get(
                                    "id"
                                )

                                if chunk_id:

                                    st.markdown(
                                        f"**Chunk ID:** "
                                        f"`{chunk_id}`"
                                    )

                                reranker_score = source.get(
                                    "reranker_score"
                                )

                                if reranker_score is not None:

                                    st.markdown(
                                        "**Reranker score:** "
                                        f"{float(reranker_score):.4f}"
                                    )

                                rrf_score = source.get(
                                    "rrf_score"
                                )

                                if rrf_score is not None:

                                    st.markdown(
                                        "**RRF score:** "
                                        f"{float(rrf_score):.6f}"
                                    )

                                if verified:

                                    st.success(
                                        "✓ Verified citation"
                                    )

                                elif unsupported:

                                    st.error(
                                        "✗ Unsupported citation"
                                    )

                    # =================================================
                    # CITATION VERIFICATION
                    # =================================================

                    if citation_verification:

                        with st.expander(
                            "🔎 Citation Verification"
                        ):

                            verified_items = (
                                citation_verification.get(
                                    "verified",
                                    [],
                                )
                            )

                            unsupported_items = (
                                citation_verification.get(
                                    "unsupported",
                                    [],
                                )
                            )

                            invalid_items = (
                                citation_verification.get(
                                    "invalid",
                                    [],
                                )
                            )

                            st.write(
                                "Verified:",
                                len(verified_items)
                            )

                            st.write(
                                "Unsupported:",
                                len(unsupported_items)
                            )

                            st.write(
                                "Invalid:",
                                len(invalid_items)
                            )

                            if verified_items:

                                st.markdown(
                                    "**Verified citations**"
                                )

                                for item in verified_items:

                                    if isinstance(
                                        item,
                                        dict,
                                    ):

                                        citation_number = item.get(
                                            "citation"
                                        )

                                        claim = item.get(
                                            "claim"
                                        )

                                        if citation_number:

                                            st.write(
                                                f"✓ [{citation_number}]"
                                            )

                                        if claim:

                                            st.caption(
                                                str(claim)
                                            )

                                    else:

                                        st.write(
                                            f"✓ {item}"
                                        )

                            if unsupported_items:

                                st.markdown(
                                    "**Unsupported citations**"
                                )

                                for item in unsupported_items:

                                    if isinstance(
                                        item,
                                        dict,
                                    ):

                                        citation_number = item.get(
                                            "citation"
                                        )

                                        st.write(
                                            f"⚠ [{citation_number}]"
                                        )

                                        claim = item.get(
                                            "claim"
                                        )

                                        if claim:

                                            st.caption(
                                                str(claim)
                                            )

                                    else:

                                        st.write(
                                            f"⚠ {item}"
                                        )

                            if invalid_items:

                                st.markdown(
                                    "**Invalid citations**"
                                )

                                for item in invalid_items:

                                    st.write(
                                        f"✗ {item}"
                                    )

                    # =================================================
                    # RETRIEVAL INFORMATION
                    # =================================================

                    with st.expander(
                        "🔍 Retrieval Information"
                    ):

                        st.write(
                            "**Retrieved chunks:**",
                            retrieved_chunks,
                        )

                        retrieval_confidence = data.get(
                            "retrieval_confidence"
                        )

                        if retrieval_confidence is not None:

                            st.write(
                                "**Retrieval confidence:**",
                                confidence_percent(
                                    retrieval_confidence
                                ),
                            )

                        citation_coverage = data.get(
                            "citation_coverage"
                        )

                        if citation_coverage is not None:

                            st.write(
                                "**Citation coverage:**",
                                confidence_percent(
                                    citation_coverage
                                ),
                            )

                        answer_completeness = data.get(
                            "answer_completeness"
                        )

                        if answer_completeness is not None:

                            st.write(
                                "**Answer completeness:**",
                                confidence_percent(
                                    answer_completeness
                                ),
                            )

                    # =================================================
                    # QUESTION
                    # =================================================

                    with st.expander(
                        "❓ Question"
                    ):

                        st.write(
                            returned_question
                        )

                    # =================================================
                    # RAW API RESPONSE
                    # =================================================

                    with st.expander(
                        "🔧 Raw API Response"
                    ):

                        st.json(data)

                # =====================================================
                # BAD REQUEST
                # =====================================================

                elif response.status_code == 400:

                    st.warning(
                        get_error_detail(response)
                    )

                # =====================================================
                # NOT READY
                # =====================================================

                elif response.status_code == 503:

                    st.error(
                        get_error_detail(response)
                    )

                # =====================================================
                # SERVER ERROR
                # =====================================================

                else:

                    st.error(
                        f"API error "
                        f"({response.status_code}): "
                        f"{get_error_detail(response)}"
                    )

            except requests.Timeout:

                st.error(
                    "The request timed out. "
                    "The RAG pipeline may be taking too long."
                )

            except requests.ConnectionError as error:

                st.error(
                    "Could not connect to the FastAPI server."
                )

                st.caption(
                    f"API URL: {API_URL}"
                )

                st.caption(
                    f"Error: {error}"
                )

            except requests.RequestException as error:

                st.error(
                    f"Request failed: {error}"
                )

            except Exception as error:

                st.error(
                    f"Streamlit error: "
                    f"{type(error).__name__}: {error}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System"
    )

    # --------------------------------------------------------
    # FastAPI
    # --------------------------------------------------------

    st.write(
        "**FastAPI:**"
    )

    st.code(
        API_URL
    )

    # --------------------------------------------------------
    # Ask endpoint
    # --------------------------------------------------------

    st.write(
        "**Ask endpoint:**"
    )

    st.code(
        "/v1/ask"
    )

    st.divider()

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.write(
        "**Hybrid retrieval pipeline:**"
    )

    st.write(
        "Dense Retrieval → BM25 → RRF → "
        "Cross-Encoder → Groq"
    )

    st.divider()

    # --------------------------------------------------------
    # System status
    # --------------------------------------------------------

    st.write(
        "**System status**"
    )

    if api_online:

        st.success(
            "🟢 RAG Ready"
        )

    else:

        st.error(
            "🔴 RAG Offline"
        )

    # --------------------------------------------------------
    # Indexed chunks
    # --------------------------------------------------------

    chunks_loaded = health_data.get(
        "chunks_loaded",
        status_data.get(
            "chunks_loaded",
            0,
        ),
    )

    st.write(
        "**Indexed Chunks**"
    )

    st.metric(
        "Chunks",
        chunks_loaded,
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_ready = health_data.get(
        "bm25_ready",
        status_data.get(
            "bm25_ready",
            False,
        ),
    )

    if bm25_ready:

        st.write(
            "BM25: 🟩 Ready"
        )

    else:

        st.write(
            "BM25: 🟥 Not Ready"
        )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranker_loaded = status_data.get(
        "reranker_loaded",
        False,
    )

    if reranker_loaded:

        st.write(
            "Reranker: 🟩 Loaded"
        )

    else:

        st.write(
            "Reranker: 🟥 Not Loaded"
        )

    st.divider()

    # --------------------------------------------------------
    # Indexed documents
    # --------------------------------------------------------

    st.write(
        "**Indexed Documents**"
    )

    if api_online:

        try:

            documents_response = requests.get(
                DOCUMENTS_ENDPOINT,
                timeout=10,
            )

            if documents_response.status_code == 200:

                documents_data = (
                    documents_response.json()
                )

                document_count = documents_data.get(
                    "count",
                    0,
                )

                st.write(
                    f"Documents: **{document_count}**"
                )

                documents = documents_data.get(
                    "documents",
                    [],
                )

                if documents:

                    with st.expander(
                        "View documents"
                    ):

                        for document in documents:

                            source = document.get(
                                "source",
                                "Unknown",
                            )

                            chunk_count = document.get(
                                "chunk_count",
                                0,
                            )

                            st.markdown(
                                f"**{source}**"
                            )

                            st.caption(
                                f"Chunks: {chunk_count}"
                            )

                            pages = document.get(
                                "pages",
                                [],
                            )

                            if pages:

                                st.caption(
                                    f"Pages: "
                                    f"{len(pages)}"
                                )

                            st.divider()

            else:

                st.caption(
                    "Could not load document list."
                )

        except requests.RequestException:

            st.caption(
                "Document information unavailable."
            )