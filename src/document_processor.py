
from pathlib import Path
from typing import List, Dict
import re
import json

import pdfplumber


def clean_text(text: str) -> str:
    """Clean extracted PDF text while preserving useful financial information."""
    if not text:
        return ""

    # Normalize whitespace but preserve line boundaries
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pages(pdf_path: str) -> List[Dict]:
    """Extract page-level text from a PDF."""
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            pages.append({
                "page": page_number,
                "text": clean_text(text)
            })

    return pages


def recursive_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[str]:
    """
    Lightweight recursive-style chunking without requiring LangChain.
    Splits on paragraphs, lines, and sentences while maintaining overlap.
    """

    if not text:
        return []

    # First split on paragraphs
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # If adding the paragraph stays within the target size
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current}\n{paragraph}".strip()
            continue

        # Save current chunk
        if current:
            chunks.append(current)

        # Handle paragraphs larger than chunk_size
        if len(paragraph) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(current) + len(sentence) + 1 <= chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)

                    # Keep oversized sentence as a chunk
                    current = sentence

            if current:
                chunks.append(current)
                current = ""
        else:
            # Add overlap from previous chunk
            if chunks and chunk_overlap > 0:
                overlap = chunks[-1][-chunk_overlap:]
                current = f"{overlap}\n{paragraph}".strip()
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def extract_tables(pdf_path: str) -> List[Dict]:
    """Extract tables from every PDF page."""
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            page_tables = page.extract_tables()

            for table_number, table in enumerate(page_tables, start=1):

                if not table:
                    continue

                cleaned_rows = []

                for row in table:
                    if not row:
                        continue

                    cleaned_row = [
                        clean_text(str(cell)) if cell is not None else ""
                        for cell in row
                    ]

                    # Ignore completely empty rows
                    if any(cell.strip() for cell in cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if cleaned_rows:
                    tables.append({
                        "page": page_number,
                        "table_number": table_number,
                        "rows": cleaned_rows
                    })

    return tables


def table_to_text(table_rows: List[List[str]]) -> str:
    """
    Convert a table into a readable text representation while
    preserving row/column relationships.
    """

    if not table_rows:
        return ""

    lines = []

    for row in table_rows:
        values = [cell.strip() for cell in row if cell.strip()]

        if values:
            lines.append(" | ".join(values))

    return "\n".join(lines)


def create_naive_chunks(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict]:
    """Create conventional text chunks."""

    pages = extract_pages(pdf_path)

    chunks = []

    chunk_id = 0

    for page_data in pages:

        page_chunks = recursive_chunks(
            page_data["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for text in page_chunks:

            if not text.strip():
                continue

            chunks.append({
                "chunk_id": f"naive_{chunk_id}",
                "document": Path(pdf_path).name,
                "page": page_data["page"],
                "chunk_type": "text",
                "text": text.strip()
            })

            chunk_id += 1

    return chunks


def create_table_aware_chunks(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict]:
    """
    Create improved chunks by preserving both page text and
    extracted financial tables.
    """

    pages = extract_pages(pdf_path)
    tables = extract_tables(pdf_path)

    chunks = []
    chunk_id = 0

    # ---------------------------------------------------------
    # PAGE TEXT
    # ---------------------------------------------------------

    for page_data in pages:

        page_chunks = recursive_chunks(
            page_data["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for text in page_chunks:

            if not text.strip():
                continue

            chunks.append({
                "chunk_id": f"table_text_{chunk_id}",
                "document": Path(pdf_path).name,
                "page": page_data["page"],
                "chunk_type": "page_text",
                "text": text.strip()
            })

            chunk_id += 1

    # ---------------------------------------------------------
    # FINANCIAL TABLES
    # ---------------------------------------------------------

    for table in tables:

        table_text = table_to_text(table["rows"])

        if not table_text.strip():
            continue

        # Add explicit metadata so the LLM knows this is a table
        structured_text = (
            f"FINANCIAL TABLE\n"
            f"Document: {Path(pdf_path).name}\n"
            f"Page: {table['page']}\n"
            f"Table: {table['table_number']}\n\n"
            f"{table_text}"
        )

        # Split only if the table is extremely large
        table_chunks = recursive_chunks(
            structured_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for text in table_chunks:

            chunks.append({
                "chunk_id": f"table_{chunk_id}",
                "document": Path(pdf_path).name,
                "page": table["page"],
                "chunk_type": "financial_table",
                "text": text.strip()
            })

            chunk_id += 1

    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Save chunks as JSON."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
