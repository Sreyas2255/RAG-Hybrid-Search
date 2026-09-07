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
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING (dark, professional theme)
# ============================================================

def inject_custom_css():

    st.markdown(
        """
        <style>

        /* ---------------------------------------------------
           Fonts + base
        --------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --accent: #6C5CE7;
            --accent-soft: rgba(108, 92, 231, 0.15);
            --accent-glow: rgba(108, 92, 231, 0.35);
            --surface: #161A23;
            --surface-2: #1D212C;
            --border: rgba(255, 255, 255, 0.08);
            --text-soft: #A6ACC0;
            --good: #33D9A2;
            --warn: #F5A623;
            --bad: #F4586B;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, rgba(108,92,231,0.10), transparent 60%),
                radial-gradient(900px 400px at 100% 0%, rgba(51,217,162,0.06), transparent 55%),
                #0E1117;
        }

        /* Hide default Streamlit chrome for a cleaner, product-like feel */
        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---------------------------------------------------
           Hero header
        --------------------------------------------------- */
        .hero-card {
            padding: 1.6rem 1.9rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(108,92,231,0.18), rgba(51,217,162,0.06));
            border: 1px solid var(--border);
            margin-bottom: 1.4rem;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #B9AFFF;
            background: var(--accent-soft);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            background: linear-gradient(90deg, #FFFFFF, #C9C3FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-sub {
            color: var(--text-soft);
            font-size: 0.98rem;
            max-width: 780px;
            line-height: 1.5;
            margin: 0;
        }

        /* ---------------------------------------------------
           Section headers
        --------------------------------------------------- */
        h2, h3 {
            font-weight: 700 !important;
        }

        /* ---------------------------------------------------
           Cards / containers with a visible border
        --------------------------------------------------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
        }

        /* ---------------------------------------------------
           Buttons
        --------------------------------------------------- */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: 1px solid var(--border);
            transition: all 0.15s ease;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #6C5CE7, #8B7BFF);
            border: none;
            box-shadow: 0 4px 18px var(--accent-glow);
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 24px var(--accent-glow);
        }

        /* ---------------------------------------------------
           Text area / inputs
        --------------------------------------------------- */
        .stTextArea textarea {
            border-radius: 14px !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            font-size: 1rem;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px var(--accent-glow) !important;
        }

        /* ---------------------------------------------------
           Metrics
        --------------------------------------------------- */
        div[data-testid="stMetric"] {
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--text-soft) !important;
        }

        /* ---------------------------------------------------
           Expanders
        --------------------------------------------------- */
        details {
            background: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        /* ---------------------------------------------------
           Badges (used for verified / unsupported / status)
        --------------------------------------------------- */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid var(--border);
        }
        .badge-good  { color: var(--good);  background: rgba(51,217,162,0.12); }
        .badge-warn  { color: var(--warn);  background: rgba(245,166,35,0.12); }
        .badge-bad   { color: var(--bad);   background: rgba(244,88,107,0.12); }
        .badge-muted { color: var(--text-soft); background: rgba(255,255,255,0.05); }

        /* ---------------------------------------------------
           Sidebar
        --------------------------------------------------- */
        section[data-testid="stSidebar"] {
            background: #0B0D13;
            border-right: 1px solid var(--border);
        }

        /* Code blocks / mono */
        code, .stCode, .stCaption {
            font-family: 'JetBrains Mono', monospace !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text, kind="muted"):
    """
    Render a small pill-shaped status badge.
    kind: "good" | "warn" | "bad" | "muted"
    """

    st.markdown(
        f'<span class="badge badge-{kind}">{text}</span>',
        unsafe_allow_html=True,
    )


inject_custom_css()


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

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">📚 Hybrid Retrieval &nbsp;•&nbsp; Dense + BM25 + Rerank</div>
        <p class="hero-title">Hi, what can I help you find?</p>
        <p class="hero-sub">
            Ask me anything about your indexed documents. I'll search dense embeddings
            and BM25 together, rerank the best matches, and answer with citations you
            can trust. If it's not in your documents, I'll say so and answer from
            general knowledge instead — clearly labeled either way.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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
                "🟢 All set — I'm online and ready for your questions."
            )

        else:

            st.warning(
                "🟡 I'm up, but still finishing my setup — "
                "give it a moment and try again."
            )

    else:

        st.warning(
            "Hmm, I got a response I wasn't expecting from the API. "
            "You may want to check the backend logs."
        )

except requests.RequestException as error:

    st.error(
        "🔴 I can't reach the backend right now."
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

with st.container(border=True):

    st.markdown("#### 💬 Ask a question")

    question = st.text_area(
        "Enter your question:",
        placeholder=(
            "e.g. What is deep learning, and how does it differ from "
            "classical machine learning?"
        ),
        height=120,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    ask_clicked = st.button(
        "🔎  Ask",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_clicked:

    if not question.strip():

        st.warning(
            "Type a question first — I'm ready when you are 🙂"
        )

    elif not api_online:

        st.error(
            "I'm not ready to answer yet — the backend isn't online."
        )

    else:

        with st.spinner(
            "Reading through your documents and thinking it over..."
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

                    grounded = bool(
                        data.get(
                            "grounded",
                            True,
                        )
                    )

                    answer_source = data.get(
                        "answer_source",
                        "documents",
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

                    st.markdown("<br>", unsafe_allow_html=True)

                    with st.container(border=True):

                        st.markdown("#### 💡 Here's what I found")

                        if (
                            grounded
                            and answer_source == "documents"
                        ):

                            badge(
                                "📚 Grounded in your documents",
                                "good",
                            )

                        else:

                            badge(
                                "🌐 General knowledge — not from your documents",
                                "warn",
                            )

                        st.write("")

                        st.write(
                            answer
                        )

                    # =================================================
                    # ANSWER QUALITY
                    # =================================================

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 📊 How confident should you be?")

                    col1, col2, col3, col4 = st.columns(4)

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

                            badge("⬆ High", "good")

                        elif label == "Medium":

                            badge("→ Medium", "warn")

                        elif label == "Low":

                            badge("⬇ Low", "bad")

                        else:

                            badge("N/A", "muted")

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

                    # -------------------------------------------------
                    # Answer source
                    # -------------------------------------------------

                    with col4:

                        st.metric(
                            "Source type",
                            (
                                "Documents"
                                if grounded
                                and answer_source == "documents"
                                else "General knowledge"
                            ),
                        )

                    # =================================================
                    # CITATIONS
                    # =================================================

                    if (
                        grounded
                        and answer_source == "documents"
                    ):

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 📚 Citations")

                        if not citations:

                            st.info(
                                "No structured citation metadata "
                                "came back with this answer."
                            )

                        else:

                            for citation_id in citations:

                                source = source_map.get(
                                    citation_id
                                )

                                with st.container(border=True):

                                    st.markdown(
                                        f"**Citation [{citation_id}]**"
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

                                            badge("✓ Verified", "good")

                                        elif unsupported:

                                            badge("✗ Unsupported", "bad")

                                        else:

                                            badge("Metadata available", "muted")

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

                                st.write("")
                    if (
                        grounded
                        and answer_source == "documents"
                    ):

                        # =================================================
                        # RETRIEVED SOURCES
                        # =================================================

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 📄 Retrieved sources")

                        if not sources:

                            st.info(
                                "No retrieved source metadata "
                                "came back with this answer."
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

                                        badge("✓ Verified citation", "good")

                                    elif unsupported:

                                        badge("✗ Unsupported citation", "bad")
                    if (
                        grounded
                        and answer_source == "documents"
                        and citation_verification
                    ):

                        # =================================================
                        # CITATION VERIFICATION
                        # =================================================

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
                    "⏱️ That took too long and timed out. "
                    "The pipeline might be under heavy load — try again in a moment."
                )

            except requests.ConnectionError as error:

                st.error(
                    "🔌 I couldn't reach the backend server."
                )

                st.caption(
                    f"API URL: {API_URL}"
                )

                st.caption(
                    f"Error: {error}"
                )

            except requests.RequestException as error:

                st.error(
                    f"Something went wrong with that request: {error}"
                )

            except Exception as error:

                st.error(
                    f"An unexpected error occurred: "
                    f"{type(error).__name__}: {error}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ System")

    # --------------------------------------------------------
    # System status
    # --------------------------------------------------------

    if api_online:

        badge("🟢 Ready to answer", "good")

    else:

        badge("🔴 Offline", "bad")

    st.write("")

    with st.container(border=True):

        # ----------------------------------------------------
        # Indexed chunks
        # ----------------------------------------------------

        chunks_loaded = health_data.get(
            "chunks_loaded",
            status_data.get(
                "chunks_loaded",
                0,
            ),
        )

        st.metric(
            "Indexed chunks",
            chunks_loaded,
        )

        # ----------------------------------------------------
        # BM25 / Reranker
        # ----------------------------------------------------

        bm25_ready = health_data.get(
            "bm25_ready",
            status_data.get(
                "bm25_ready",
                False,
            ),
        )

        reranker_loaded = status_data.get(
            "reranker_loaded",
            False,
        )

        status_col1, status_col2 = st.columns(2)

        with status_col1:

            st.caption("BM25")

            if bm25_ready:
                badge("Ready", "good")
            else:
                badge("Not ready", "bad")

        with status_col2:

            st.caption("Reranker")

            if reranker_loaded:
                badge("Loaded", "good")
            else:
                badge("Not loaded", "bad")

    st.write("")

    with st.expander("🔧 Connection details"):

        st.caption("FastAPI base URL")
        st.code(API_URL)

        st.caption("Ask endpoint")
        st.code("/v1/ask")

        st.caption("Pipeline")
        st.write(
            "Dense Retrieval → BM25 → RRF → "
            "Cross-Encoder → Groq"
        )

    st.divider()

    # --------------------------------------------------------
    # Indexed documents
    # --------------------------------------------------------

    st.markdown("**📁 Indexed documents**")

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