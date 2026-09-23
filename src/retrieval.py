
from pathlib import Path
from typing import List, Dict, Tuple
import json

import numpy as np
import faiss


class FAISSRetriever:
    """
    FAISS-based semantic retriever for FinAudit RAGBench.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        """
        Create embeddings and build a FAISS inner-product index.
        Embeddings are normalized, making inner product equivalent
        to cosine similarity.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = chunks

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
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
