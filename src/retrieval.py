from pathlib import Path
from typing import List, Dict
import json
import re

import numpy as np
import faiss


class FAISSRetriever:
    """
    FAISS-based semantic retriever for FinAudit RAGBench.

    Uses dense semantic similarity with a lightweight
    lexical reranking signal.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: List[Dict]):
        """
        Create embeddings and build a FAISS inner-product index.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = chunks

        texts = [
            str(chunk.get("text", ""))
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return embeddings

    @staticmethod
    def _tokenize(text: str) -> set:
        """
        Convert text into normalized word tokens.
        """

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower()
            )
        )

    @staticmethod
    def _query_terms(query: str) -> set:
        """
        Extract meaningful terms from the query.
        """

        stopwords = {
            "what",
            "was",
            "were",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "for",
            "in",
            "on",
            "to",
            "and",
            "or",
            "with",
            "from",
            "how",
            "much",
            "did",
            "does",
            "company"
        }

        tokens = FAISSRetriever._tokenize(query)

        return {
            token
            for token in tokens
            if token not in stopwords
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using semantic similarity
        followed by lightweight lexical reranking.
        """

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        if not query or not query.strip():
            return []

        total_chunks = len(self.chunks)

        candidate_k = min(
            max(top_k * 4, 10),
            total_chunks
        )

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            candidate_k
        )

        query_terms = self._query_terms(query)

        candidates = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            chunk = self.chunks[index].copy()

            text = str(
                chunk.get("text", "")
            )

            text_terms = self._tokenize(text)

            if query_terms:
                overlap = (
                    len(query_terms & text_terms)
                    / len(query_terms)
                )
            else:
                overlap = 0.0

            semantic_score = float(score)

            final_score = (
                0.85 * semantic_score
                + 0.15 * overlap
            )

            chunk["score"] = semantic_score

            chunk["lexical_overlap"] = float(
                overlap
            )

            chunk["rerank_score"] = float(
                final_score
            )

            candidates.append(chunk)

        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:top_k]

    def save_index(
        self,
        index_path: str,
        chunks_path: str
    ):
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:
            raise RuntimeError(
                "Cannot save an empty index."
            )

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.chunks,
                f,
                indent=2,
                ensure_ascii=False
            )


def load_chunks(path: str) -> List[Dict]:
    """
    Load chunk metadata from JSON.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = chunks

        texts = [
            str(chunk.get("text", ""))
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)

        self.embeddings = embeddings

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return embeddings

    @staticmethod
    def _tokenize(text: str) -> set:
        """
        Convert text into normalized word tokens.
        """

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower()
            )
        )

    @staticmethod
    def _query_terms(query: str) -> set:
        """
        Extract useful terms from the query.

        Common stopwords are removed so that important
        financial terms receive more weight.
        """

        stopwords = {
            "what",
            "was",
            "were",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "for",
            "in",
            "on",
            "to",
            "and",
            "or",
            "with",
            "from",
            "how",
            "much",
            "did",
            "does",
            "company"
        }

        tokens = FAISSRetriever._tokenize(query)

        return {
            token
            for token in tokens
            if token not in stopwords
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using semantic similarity
        followed by lightweight lexical reranking.

        This improves retrieval for questions containing
        important financial terms, fiscal years, and
        numerical concepts without turning the retriever
        into a table parser.
        """

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        if not query or not query.strip():
            return []

        total_chunks = len(self.chunks)

        # Retrieve a larger candidate pool first.
        candidate_k = min(
            max(top_k * 4, 10),
            total_chunks
        )

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            candidate_k
        )

        query_terms = self._query_terms(query)

        candidates = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            chunk = self.chunks[index].copy()

            text = str(
                chunk.get("text", "")
            )

            text_terms = self._tokenize(text)

            if query_terms:
                overlap = (
                    len(query_terms & text_terms)
                    / len(query_terms)
                )
            else:
                overlap = 0.0

            semantic_score = float(score)

            # Mostly semantic retrieval, with a small
            # lexical signal to preserve important terms.
            final_score = (
                0.85 * semantic_score
                + 0.15 * overlap
            )

            chunk["score"] = semantic_score
            chunk["lexical_overlap"] = float(
                overlap
            )
            chunk["rerank_score"] = float(
                final_score
            )

            candidates.append(chunk)

        # Highest combined score first.
        candidates.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return candidates[:top_k]

    def save_index(embeddings = self.embedding_model.encode)
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve the top-k most similar chunks.
        """

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks))
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            chunk = self.chunks[index].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    def save_index(
        self,
        index_path: str,
        chunks_path: str
    ):
        """Save FAISS index and chunk metadata."""

        if self.index is None:
            raise RuntimeError(
                "Cannot save an empty index."
            )

        index_path = Path(index_path)
        chunks_path = Path(chunks_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        chunks_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.chunks,
                f,
                indent=2,
                ensure_ascii=False
            )


def load_chunks(path: str) -> List[Dict]:
    """Load chunk metadata from JSON."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)
