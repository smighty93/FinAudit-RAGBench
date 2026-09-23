# FinAudit RAGBench

## An Automated Benchmarking Platform for Evaluating Financial Retrieval-Augmented Generation Systems

FinAudit RAGBench is a financial-domain benchmarking platform designed to evaluate the reliability of Retrieval-Augmented Generation (RAG) systems when answering questions from financial documents.

The project compares two RAG approaches:

1. **Naive RAG**
   - Conventional document text extraction
   - Text chunking
   - Embedding generation
   - FAISS vector retrieval
   - LLM-based answer generation

2. **Table-Aware RAG**
   - Financial document processing with improved handling of structured/table-oriented information
   - Embedding generation
   - FAISS vector retrieval
   - LLM-based answer generation

Both approaches were evaluated using the same financial document and controlled benchmark questions.

## Business Problem

Financial reports contain large amounts of numerical information, financial tables, reporting periods, footnotes, and repeated information across different years. Retrieval errors can cause a RAG system to provide incomplete or incorrect financial answers.

FinAudit RAGBench evaluates retrieval and answer-generation performance to identify these failure cases.

## Evaluation

The benchmark evaluates:

- Numerical Accuracy
- Source-Page Recall@5
- Mean Reciprocal Rank (MRR)
- Average Retrieval/Generation Latency
- Failure cases

The displayed results are generated from the executed benchmark and are not manually fabricated.

## Dataset

The current demonstration uses an Apple financial report containing financial information for fiscal years 2025 and 2024.

## Application

The Streamlit dashboard provides:

- Project overview
- Benchmark results
- RAG comparison
- Query-level analysis
- Retrieved information
- Generated answers
- Numerical accuracy
- Failure analysis
- Evaluation visualizations

## Technology Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyPDF
- pdfplumber
- pandas
- NumPy
- scikit-learn
- Plotly
- Google Gemini API

## Project Structure

```text
FinAudit_RAGBench/
│
├── app.py
├── benchmark_questions.json
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
└── src/
