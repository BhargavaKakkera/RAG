
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class ThresholdRetriever:
    """Semantic retriever that filters out low-confidence Chroma matches."""

    vectorstore: object
    k: int
    similarity_threshold: float

    def retrieve(self, query: str) -> list[Document]:
        try:
            scored_docs = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.k,
            )
        except Exception as exc:
            raise RuntimeError(f"Retrieval failed: {exc}") from exc

        filtered: list[Document] = []
        for document, score in scored_docs:
            if score >= self.similarity_threshold:
                document.metadata["similarity_score"] = round(float(score), 4)
                filtered.append(document)

        return filtered

