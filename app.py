import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


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
RAW_RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"


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
# SIDEBAR
# ============================================================

st.sidebar.title("FinAudit RAGBench")

st.sidebar.markdown(
    """
**Financial RAG Benchmarking Platform**

Compare Naive RAG and Table-Aware RAG
on financial question answering.
"""
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Benchmark Results",
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
FinAudit RAGBench evaluates the reliability of RAG systems
when answering questions from financial documents.

The current experiment compares two approaches:

- **Naive RAG**
- **Table-Aware RAG**

Both approaches use the same financial benchmark and are
evaluated using numerical accuracy, source-page recall,
MRR, and latency.
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
Chunking
        ↓
Embeddings
        ↓
FAISS Vector Search
        ↓
Context Retrieval
        ↓
LLM
        ↓
Generated Answer
        ↓
Evaluation
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
            "Document",
            "Apple Financial Report"
        )

    st.divider()

    st.header("Implemented GenAI Components")

    components = pd.DataFrame({
        "Component": [
            "Retrieval-Augmented Generation",
            "Text Embeddings",
            "FAISS Vector Search",
            "Prompt-based LLM Generation",
            "Financial Document Processing",
            "RAG Evaluation",
            "Streamlit Visualization"
        ],
        "Status": [
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented"
        ]
    })

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

        # Numerical accuracy

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

        # MRR

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

        # Latency

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
# QUERY ANALYSIS
# ============================================================

elif page == "Query Analysis":

    st.title("🔎 Query Analysis")

    if benchmark_df.empty:

        st.error("Benchmark results not found.")

    else:

        selected_id = st.selectbox(
            "Select Question",
            benchmark_df["Question ID"].tolist()
        )

        selected = benchmark_df[
            benchmark_df["Question ID"] == selected_id
        ].iloc[0]

        st.header(selected["Question"])

        st.markdown(
            f"**Reference Answer:** {selected['Reference Answer']}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Naive RAG")

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

            st.subheader("Table-Aware RAG")

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
                    f"**Reference:** {row['Reference Answer']}"
                )

                st.subheader("Naive RAG")

                st.write(
                    row["Naive Answer"]
                )

                st.write(
                    f"Result: {row['Naive Result']}"
                )

                st.subheader("Table-Aware RAG")

                st.write(
                    row["Table-Aware Answer"]
                )

                st.write(
                    f"Result: {row['Table-Aware Result']}"
                )

        st.divider()

        st.subheader("Observed Failure Pattern")

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
