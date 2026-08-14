import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="centered",
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Hybrid RAG Assistant")

st.write(
    "Ask questions about the documents indexed by the RAG system."
)


# ============================================================
# API HEALTH CHECK
# ============================================================

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    if health_response.status_code == 200:

        health_data = health_response.json()

        if health_data.get("bm25_ready"):

            st.success("RAG API is online and ready.")

        else:

            st.warning("RAG API is running but not ready.")

    else:

        st.warning("RAG API returned an unexpected response.")

except requests.RequestException:

    st.error(
        "Could not connect to the FastAPI server. "
        "Make sure FastAPI is running."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Enter your question:",
    placeholder="Example: What is a machine learning algorithm?",
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "Ask Question",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question.strip()
                    },
                    timeout=120,
                )

                # ------------------------------------------------
                # Successful response
                # ------------------------------------------------

                if response.status_code == 200:

                    data = response.json()

                    st.subheader("Answer")

                    st.write(
                        data.get(
                            "answer",
                            "No answer returned.",
                        )
                    )

                    # ------------------------------------------------
                    # Question
                    # ------------------------------------------------

                    with st.expander(
                        "Question"
                    ):

                        st.write(
                            data.get(
                                "question",
                                question,
                            )
                        )

                    # ------------------------------------------------
                    # Raw API response
                    # ------------------------------------------------

                    with st.expander(
                        "API Response"
                    ):

                        st.json(data)

                # ------------------------------------------------
                # Bad request
                # ------------------------------------------------

                elif response.status_code == 400:

                    error_data = response.json()

                    st.warning(
                        error_data.get(
                            "detail",
                            "Invalid request.",
                        )
                    )

                # ------------------------------------------------
                # Service unavailable
                # ------------------------------------------------

                elif response.status_code == 503:

                    error_data = response.json()

                    st.error(
                        error_data.get(
                            "detail",
                            "RAG system is not ready.",
                        )
                    )

                # ------------------------------------------------
                # Server error
                # ------------------------------------------------

                else:

                    try:

                        error_data = response.json()

                        detail = error_data.get(
                            "detail",
                            "Unknown API error.",
                        )

                    except ValueError:

                        detail = response.text

                    st.error(
                        f"API error ({response.status_code}): "
                        f"{detail}"
                    )

            except requests.Timeout:

                st.error(
                    "The request timed out. "
                    "The RAG pipeline may be taking too long."
                )

            except requests.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server."
                )

            except requests.RequestException as error:

                st.error(
                    f"Request failed: {error}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.write(
        "**FastAPI:**"
    )

    st.code(
        API_URL
    )

    st.write(
        "**Endpoint:**"
    )

    st.code(
        "/ask"
    )

    st.divider()

    st.write(
        "Hybrid retrieval pipeline:"
    )

    st.write(
        "Dense Retrieval → BM25 → RRF → "
        "Cross-Encoder → Groq"
    )