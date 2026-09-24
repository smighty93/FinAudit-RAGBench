import sys
import time
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.document_processor import (
    create_naive_chunks,
    create_table_aware_chunks
)

from src.retrieval import FAISSRetriever

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinAudit RAGBench",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

METRICS_FILE = RESULTS_DIR / "final_metrics.csv"
BENCHMARK_FILE = RESULTS_DIR / "final_benchmark_results.csv"
FAILURE_FILE = RESULTS_DIR / "failure_analysis.csv"


# ============================================================
# LOAD EXISTING BENCHMARK DATA
# ============================================================

@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return pd.DataFrame()


@st.cache_data
def load_benchmark():
    if BENCHMARK_FILE.exists():
        return pd.read_csv(BENCHMARK_FILE)
    return pd.DataFrame()


@st.cache_data
def load_failures():
    if FAILURE_FILE.exists():
        return pd.read_csv(FAILURE_FILE)
    return pd.DataFrame()


metrics_df = load_metrics()
benchmark_df = load_benchmark()
failure_df = load_failures()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def load_gemini_client():

    try:

        api_key = st.secrets.get("GOOGLE_API_KEY")

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None

# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    client,
    question,
    retrieved_chunks,
    approach_name
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}
Document: {chunk.get("document", "Unknown")}
Page: {chunk.get("page", "Unknown")}
Chunk Type: {chunk.get("chunk_type", "Unknown")}
Retrieval Score: {chunk.get("score", 0):.4f}

{chunk.get("text", "")}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a financial document question-answering assistant
inside the FinAudit RAGBench evaluation platform.

RAG APPROACH:
{approach_name}

Answer the user's question using ONLY the supplied financial
document context.

RULES:

1. Do not use outside knowledge.
2. Do not invent financial values.
3. Preserve financial units exactly.
4. Pay close attention to fiscal years and reporting periods.
5. If the answer cannot be established from the supplied
   context, say:
   "The provided context does not contain enough information
   to answer this question."
6. Keep the answer concise.
7. Clearly state the financial value when available.
8. Identify the supporting page number or page numbers.
9. Do not create a reference answer that is not present in
   the supplied document.

USER QUESTION:
{question}

RETRIEVED FINANCIAL CONTEXT:
{context}

ANSWER:
"""

    max_attempts = 5

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            if not response or not response.text:
                return "Gemini returned an empty response."

            return response.text.strip()

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                if attempt < max_attempts - 1:

                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                    continue

            return (
                f"Gemini API error: "
                f"{type(e).__name__}: {error_message}"
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 FinAudit RAGBench")

st.sidebar.markdown(
    """
**Financial RAG Benchmarking Platform**

Evaluate financial RAG systems and
perform interactive document-based
financial question answering.
"""
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Benchmark Results",
        "Interactive Q&A",
        "Query Analysis",
        "Failure Analysis"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("📊 FinAudit RAGBench")

    st.subheader(
        "Automated Benchmarking Platform for Financial "
        "Retrieval-Augmented Generation Systems"
    )

    st.markdown(
        """
FinAudit RAGBench evaluates the reliability of
Retrieval-Augmented Generation systems when answering
questions from financial documents.

The platform supports two complementary workflows:

**Benchmark Mode**
- Controlled financial benchmark
- Reference answers
- Numerical accuracy
- Source-page recall
- MRR
- Latency
- Failure analysis

**Interactive Q&A Mode**
- Upload a financial PDF
- Ask financial questions
- Compare Naive RAG and Table-Aware RAG
- Inspect retrieved evidence and source pages
"""
    )

    st.divider()

    st.header("Business Problem")

    st.write(
        """
Financial reports contain large amounts of numerical data,
financial tables, reporting periods, footnotes, and repeated
information.

Poor chunking or retrieval can provide incomplete or irrelevant
context to an LLM, resulting in unreliable financial answers.

FinAudit RAGBench provides a controlled environment for
measuring these retrieval and generation differences.
"""
    )

    st.header("System Workflow")

    st.code(
        """
Financial Document
        ↓
Document Processing
        ↓
Naive / Table-Aware Chunking
        ↓
Sentence Embeddings
        ↓
FAISS Vector Search
        ↓
Context Retrieval
        ↓
Gemini LLM
        ↓
Generated Answer
        ↓
Evidence / Evaluation
        ↓
Streamlit Dashboard
""",
        language="text"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Benchmark Questions",
            len(benchmark_df)
        )

    with col2:
        st.metric(
            "RAG Approaches",
            "2"
        )

    with col3:
        st.metric(
            "Benchmark Document",
            "Apple Financial Report"
        )

    st.divider()

    st.header("Implemented GenAI Components")

    components = pd.DataFrame([
        {
            "Component": "Retrieval-Augmented Generation",
            "Status": "Implemented"
        },
        {
            "Component": "Text Embeddings",
            "Status": "Implemented"
        },
        {
            "Component": "FAISS Vector Search",
            "Status": "Implemented"
        },
        {
            "Component": "Gemini LLM Generation",
            "Status": "Implemented"
        },
        {
            "Component": "Financial Document Processing",
            "Status": "Implemented"
        },
        {
            "Component": "RAG Evaluation",
            "Status": "Implemented"
        },
        {
            "Component": "Interactive Financial Q&A",
            "Status": "Implemented"
        },
        {
            "Component": "Streamlit Visualization",
            "Status": "Implemented"
        }
    ])

    st.dataframe(
        components,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BENCHMARK RESULTS
# ============================================================

elif page == "Benchmark Results":

    st.title("📈 Benchmark Results")

    if metrics_df.empty:

        st.error("Metrics file not found.")

    else:

        metric_index = metrics_df.set_index("Metric")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Naive Numerical Accuracy",
                f"{metric_index.loc['Numerical Accuracy', 'Naive RAG']:.0%}"
            )

            st.metric(
                "Naive Source Recall@5",
                f"{metric_index.loc['Source-Page Recall@5', 'Naive RAG']:.0%}"
            )

        with col2:

            st.metric(
                "Table-Aware Numerical Accuracy",
                f"{metric_index.loc['Numerical Accuracy', 'Table-Aware RAG']:.0%}"
            )

            st.metric(
                "Table-Aware Source Recall@5",
                f"{metric_index.loc['Source-Page Recall@5', 'Table-Aware RAG']:.0%}"
            )

        st.divider()

        accuracy_df = pd.DataFrame({
            "Approach": [
                "Naive RAG",
                "Table-Aware RAG"
            ],
            "Accuracy": [
                metric_index.loc[
                    "Numerical Accuracy",
                    "Naive RAG"
                ] * 100,
                metric_index.loc[
                    "Numerical Accuracy",
                    "Table-Aware RAG"
                ] * 100
            ]
        })

        fig = px.bar(
            accuracy_df,
            x="Approach",
            y="Accuracy",
            title="Numerical Accuracy",
            range_y=[0, 100],
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        mrr_df = pd.DataFrame({
            "Approach": [
                "Naive RAG",
                "Table-Aware RAG"
            ],
            "MRR": [
                metric_index.loc["MRR", "Naive RAG"],
                metric_index.loc["MRR", "Table-Aware RAG"]
            ]
        })

        fig = px.bar(
            mrr_df,
            x="Approach",
            y="MRR",
            title="Mean Reciprocal Rank",
            range_y=[0, 1],
            text_auto=".3f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        latency_df = pd.DataFrame({
            "Approach": [
                "Naive RAG",
                "Table-Aware RAG"
            ],
            "Latency": [
                metric_index.loc[
                    "Average Latency (seconds)",
                    "Naive RAG"
                ],
                metric_index.loc[
                    "Average Latency (seconds)",
                    "Table-Aware RAG"
                ]
            ]
        })

        fig = px.bar(
            latency_df,
            x="Approach",
            y="Latency",
            title="Average Latency",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Overall Metrics")

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Question-Level Results")

        if not benchmark_df.empty:

            columns = [
                "Question ID",
                "Category",
                "Question",
                "Naive Numerical Accuracy",
                "Table-Aware Numerical Accuracy",
                "Naive MRR",
                "Table-Aware MRR"
            ]

            columns = [
                c for c in columns
                if c in benchmark_df.columns
            ]

            st.dataframe(
                benchmark_df[columns],
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# INTERACTIVE Q&A
# ============================================================

elif page == "Interactive Q&A":

    st.title("🔎 Interactive Financial Q&A")

    st.markdown(
        """
Upload a financial PDF and ask questions about its contents.

The system runs the same question through:

- **Naive RAG**
- **Table-Aware RAG**

The generated answers can then be compared together with
the retrieved evidence and source pages.
"""
    )

    uploaded_file = st.file_uploader(
        "Upload a financial PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        question = st.text_area(
            "Enter your financial question",
            placeholder=(
                "Example: What was the company's total revenue "
                "for fiscal year 2025?"
            )
        )

        top_k = st.slider(
            "Number of retrieved chunks",
            min_value=3,
            max_value=10,
            value=5
        )

        if st.button(
            "🚀 Analyze Question",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )
                st.stop()

            client = load_gemini_client()

            if client is None:

                st.error(
                    "GOOGLE_API_KEY was not found. "
                    "Configure the Colab secret before using "
                    "Interactive Q&A."
                )
                st.stop()

            embedding_model = load_embedding_model()

            with tempfile.TemporaryDirectory() as temp_dir:

                pdf_path = Path(temp_dir) / uploaded_file.name

                pdf_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                progress = st.progress(0)

                st.info(
                    "Extracting document and creating RAG indexes..."
                )

                naive_chunks = create_naive_chunks(
                    str(pdf_path)
                )

                progress.progress(25)

                table_chunks = create_table_aware_chunks(
                    str(pdf_path)
                )

                progress.progress(50)

                naive_retriever = FAISSRetriever(
                    embedding_model
                )

                naive_retriever.build_index(
                    naive_chunks
                )

                progress.progress(70)

                table_retriever = FAISSRetriever(
                    embedding_model
                )

                table_retriever.build_index(
                    table_chunks
                )

                progress.progress(85)

                # ------------------------------------------------
                # NAIVE RAG
                # ------------------------------------------------

                naive_start = time.perf_counter()

                naive_results = naive_retriever.retrieve(
                    question,
                    top_k=top_k
                )

                naive_answer = generate_answer(
                    client,
                    question,
                    naive_results,
                    "Naive RAG"
                )

                naive_latency = (
                    time.perf_counter()
                    - naive_start
                )

                # ------------------------------------------------
                # TABLE-AWARE RAG
                # ------------------------------------------------

                table_start = time.perf_counter()

                table_results = table_retriever.retrieve(
                    question,
                    top_k=top_k
                )

                table_answer = generate_answer(
                    client,
                    question,
                    table_results,
                    "Table-Aware RAG"
                )

                table_latency = (
                    time.perf_counter()
                    - table_start
                )

                progress.progress(100)

            st.success(
                "Analysis completed."
            )

            st.divider()

            col1, col2 = st.columns(2)

            # ----------------------------------------------------
            # NAIVE RESULT
            # ----------------------------------------------------

            with col1:

                st.subheader("Naive RAG")

                st.write(
                    naive_answer
                )

                st.metric(
                    "Latency",
                    f"{naive_latency:.2f} seconds"
                )

                st.markdown(
                    "**Retrieved Source Pages**"
                )

                naive_pages = sorted(
                    set(
                        str(chunk["page"])
                        for chunk in naive_results
                    )
                )

                st.write(
                    ", ".join(naive_pages)
                )

                with st.expander(
                    "View Retrieved Evidence"
                ):

                    for i, chunk in enumerate(
                        naive_results,
                        start=1
                    ):

                        st.markdown(
                            f"""
**Source {i}**

Page: {chunk["page"]}

Chunk Type: {chunk["chunk_type"]}

Retrieval Score:
`{chunk["score"]:.4f}`
"""
                        )

                        st.text(
                            chunk["text"]
                        )

                        st.divider()

            # ----------------------------------------------------
            # TABLE-AWARE RESULT
            # ----------------------------------------------------

            with col2:

                st.subheader("Table-Aware RAG")

                st.write(
                    table_answer
                )

                st.metric(
                    "Latency",
                    f"{table_latency:.2f} seconds"
                )

                st.markdown(
                    "**Retrieved Source Pages**"
                )

                table_pages = sorted(
                    set(
                        str(chunk["page"])
                        for chunk in table_results
                    )
                )

                st.write(
                    ", ".join(table_pages)
                )

                with st.expander(
                    "View Retrieved Evidence"
                ):

                    for i, chunk in enumerate(
                        table_results,
                        start=1
                    ):

                        st.markdown(
                            f"""
**Source {i}**

Page: {chunk["page"]}

Chunk Type: {chunk["chunk_type"]}

Retrieval Score:
`{chunk["score"]:.4f}`
"""
                        )

                        st.text(
                            chunk["text"]
                        )

                        st.divider()

            st.divider()

            st.subheader(
                "Evaluation Interpretation"
            )

            st.info(
                """
Interactive Q&A does not calculate numerical accuracy because
the uploaded document does not automatically provide a verified
reference answer for the user's question.

Instead, this mode exposes the generated answer, retrieved
evidence, source pages, retrieval scores, and latency.

Formal numerical accuracy remains available in Benchmark Mode,
where verified reference answers are available.
"""
            )


# ============================================================
# QUERY ANALYSIS
# ============================================================

elif page == "Query Analysis":

    st.title("🔎 Benchmark Query Analysis")

    if benchmark_df.empty:

        st.error(
            "Benchmark results not found."
        )

    else:

        selected_id = st.selectbox(
            "Select Question",
            benchmark_df["Question ID"].tolist()
        )

        selected = benchmark_df[
            benchmark_df["Question ID"] == selected_id
        ].iloc[0]

        st.header(
            selected["Question"]
        )

        st.markdown(
            f"**Reference Answer:** "
            f"{selected['Reference Answer']}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Naive RAG"
            )

            st.write(
                selected["Naive Answer"]
            )

            st.metric(
                "Numerical Accuracy",
                f"{selected['Naive Numerical Accuracy']:.0%}"
            )

            st.metric(
                "MRR",
                f"{selected['Naive MRR']:.3f}"
            )

            st.metric(
                "Latency",
                f"{selected['Naive Latency']:.2f} seconds"
            )

        with col2:

            st.subheader(
                "Table-Aware RAG"
            )

            st.write(
                selected["Table-Aware Answer"]
            )

            st.metric(
                "Numerical Accuracy",
                f"{selected['Table-Aware Numerical Accuracy']:.0%}"
            )

            st.metric(
                "MRR",
                f"{selected['Table-Aware MRR']:.3f}"
            )

            st.metric(
                "Latency",
                f"{selected['Table-Aware Latency']:.2f} seconds"
            )


# ============================================================
# FAILURE ANALYSIS
# ============================================================

elif page == "Failure Analysis":

    st.title("⚠️ Failure Analysis")

    if failure_df.empty:

        st.success(
            "No numerical-accuracy failures were recorded."
        )

    else:

        st.metric(
            "Failure Cases",
            len(failure_df)
        )

        for _, row in failure_df.iterrows():

            with st.expander(
                f"{row['Question ID']} — {row['Question']}"
            ):

                st.markdown(
                    f"**Reference:** "
                    f"{row['Reference Answer']}"
                )

                st.subheader(
                    "Naive RAG"
                )

                st.write(
                    row["Naive Answer"]
                )

                st.write(
                    f"Result: {row['Naive Result']}"
                )

                st.subheader(
                    "Table-Aware RAG"
                )

                st.write(
                    row["Table-Aware Answer"]
                )

                st.write(
                    f"Result: {row['Table-Aware Result']}"
                )

        st.divider()

        st.subheader(
            "Observed Failure Pattern"
        )

        st.info(
            """
The benchmark demonstrates that retrieving the correct source
page does not always guarantee a correct generated answer.

Some questions failed because the retrieved context did not
contain sufficient information for the LLM to produce the
expected financial value.

These cases are retained as part of the benchmark rather than
being removed or artificially corrected.
"""
        )
