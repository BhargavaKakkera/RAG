from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from utils.timing import log_timing

logger = logging.getLogger(__name__)


def adaptive_retrieval_k(question: str, max_k: int) -> int:
    """Choose retrieval K based on question complexity, capped by user max K."""

    q = question.strip().lower()
    comparison_markers = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "versus",
        " vs ",
        " vs.",
        "contrast",
        "similarities",
    ]
    complex_markers = [
        "explain in detail",
        "walk me through",
        "step by step",
        "how does",
        "why does",
        "describe in detail",
        " elaborate ",
    ]
    simple_patterns = [
        r"^what is\b",
        r"^who is\b",
        r"^when\b",
        r"^where\b",
        r"^define\b",
        r"^which\b",
    ]

    if any(marker in q for marker in comparison_markers):
        target = 8
    elif any(marker in q for marker in complex_markers):
        target = 10
    elif any(re.search(pattern, q) for pattern in simple_patterns):
        target = 3
    else:
        target = 5

    return min(max(target, 1), max_k)


@dataclass
class ThresholdRetriever:
    """Semantic retriever with threshold filtering and graceful fallback."""

    vectorstore: object
    k: int
    similarity_threshold: float

    def _search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def _apply_threshold(
        self,
        scored_docs: list[tuple[Document, float]],
        threshold: float,
    ) -> list[Document]:
        filtered: list[Document] = []
        for document, score in scored_docs:
            relevance = float(score)
            if relevance >= threshold:
                document.metadata["similarity_score"] = round(relevance, 4)
                filtered.append(document)
        return filtered

    def _best_effort(
        self,
        scored_docs: list[tuple[Document, float]],
        limit: int,
    ) -> list[Document]:
        documents: list[Document] = []
        for document, score in scored_docs[:limit]:
            document.metadata["similarity_score"] = round(float(score), 4)
            documents.append(document)
        return documents

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        effective_k = k or self.k
        relaxed_threshold = max(self.similarity_threshold - 0.15, 0.0)

        with log_timing("retrieval"):
            try:
                with log_timing("retrieval_vector_search", k=effective_k):
                    scored_docs = self._search(query, effective_k)
            except Exception as exc:
                raise RuntimeError(f"Retrieval failed: {exc}") from exc

            if not scored_docs:
                logger.info("Retrieval returned no candidates for query.")
                return []

            with log_timing(
                "retrieval_threshold_filtering",
                threshold=self.similarity_threshold,
                candidates=len(scored_docs),
            ):
                filtered = self._apply_threshold(scored_docs, self.similarity_threshold)
            if filtered:
                logger.info(
                    "Retrieval threshold satisfied (k=%s, threshold=%.2f, hits=%s).",
                    effective_k,
                    self.similarity_threshold,
                    len(filtered),
                )
                return filtered

            with log_timing(
                "retrieval_relaxed_filtering",
                relaxed_threshold=round(relaxed_threshold, 2),
                candidates=len(scored_docs),
            ):
                relaxed = self._apply_threshold(scored_docs, relaxed_threshold)
            if relaxed:
                logger.info(
                    "Retrieval fallback used relaxed threshold %.2f (hits=%s).",
                    relaxed_threshold,
                    len(relaxed),
                )
                return relaxed

            logger.info(
                "Retrieval fallback returning top-%s results without threshold.",
                effective_k,
            )
            return self._best_effort(scored_docs, effective_k)
