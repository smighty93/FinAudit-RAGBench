
from typing import List, Dict
from google import genai


class RAGEngine:
    """
    Generic RAG engine for FinAudit RAGBench.

    The retriever determines which financial context is supplied
    to the LLM. The same generation model and prompt structure
    are used for both RAG approaches.
    """

    def __init__(
        self,
        retriever,
        client,
        model_name: str,
        approach_name: str
    ):
        self.retriever = retriever
        self.client = client
        self.model_name = model_name
        self.approach_name = approach_name

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:

        context_parts = []

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):
            context_parts.append(
                f"""
SOURCE {i}
Document: {chunk.get('document', 'Unknown')}
Page: {chunk.get('page', 'Unknown')}
Chunk Type: {chunk.get('chunk_type', 'Unknown')}
Retrieval Score: {chunk.get('score', 0):.4f}

{chunk.get('text', '')}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are a financial document question-answering assistant.

Answer the user's question using ONLY the supplied financial
document context.

IMPORTANT RULES:
1. Do not invent financial values.
2. Do not use outside knowledge.
3. If the context does not contain enough information, say:
   "The provided context does not contain enough information
   to answer this question."
4. Preserve financial units exactly.
5. Pay close attention to reporting periods and years.
6. For numerical questions, identify the relevant year and
   value explicitly.
7. Give a concise answer.
8. After the answer, provide the supporting source page(s).

USER QUESTION:
{question}

RETRIEVED FINANCIAL CONTEXT:
{context}

ANSWER:
"""

        return prompt

    def answer(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict:

        retrieved = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        prompt = self.build_prompt(
            question,
            retrieved
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        answer_text = response.text

        return {
            "approach": self.approach_name,
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved,
            "model": self.model_name
        }
