import streamlit as st
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 Multimodal RAG")
st.caption(
    "Document ingestion, hybrid retrieval, reranking and multimodal generation"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "xlsx",
            "pptx"
        ]
    )

    ingest_button = st.button(
        "📥 Ingest Document",
        use_container_width=True
    )

    st.divider()

    st.header("Search Settings")

    search_type = st.selectbox(
        "Search Type",
        [
            "all",
            "text",
            "table",
            "image"
        ]
    )

    figure_number = st.number_input(
        "Figure Number (optional)",
        min_value=1,
        value=None,
        step=1
    )


# ============================================================
# DOCUMENT INGESTION
# ============================================================

if ingest_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a document first."
        )

    else:

        with st.spinner("Ingesting document..."):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files=files,
                    timeout=300
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "Document ingested successfully!"
                    )

                    st.json(result)

                else:

                    st.error(
                        f"Ingestion failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Make sure the backend is running on port 8000."
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


# ============================================================
# QUESTION
# ============================================================

st.header("Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder="Example: What does Figure 2 show?",
    height=100
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True
)


# ============================================================
# QUERY
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                payload = {
                    "question": question,
                    "search_type": search_type,
                    "figure_number": figure_number
                }

                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=300
                )

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Answer")

                    st.write(
                        result.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                    st.divider()

                    st.subheader("Query Details")

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Search Type:** "
                            f"{result.get('search_type')}"
                        )

                    with col2:

                        st.write(
                            f"**Figure Number:** "
                            f"{result.get('figure_number')}"
                        )

                else:

                    st.error(
                        f"Query failed: {response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Make sure the backend is running on port 8000."
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multimodal RAG • FastAPI • ChromaDB • BM25 • RRF • "
    "Reranker • Qwen2.5-VL"
)