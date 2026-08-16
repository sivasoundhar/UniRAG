"""
BM25 keyword search over the same corpus Chroma holds.

rank_bm25 has no persistence and no notion of "the corpus" — it just scores
whatever token lists it was built with. So this module doesn't own a corpus
of its own; a caller builds it from app.retrieval.vector_store.get_all_documents()
so BM25 and dense search are always ranking the same chunks, identified by
the same chunk_id in metadata. This is what lets app.retrieval.hybrid_fusion
merge the two rankings without any text-matching guesswork.
"""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config.settings import settings


def _tokenize(text: str) -> list[str]:
    """Lowercase word split — BM25 only needs term overlap, not linguistics."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Takes: nothing at construction — call build() before search().
    Returns: an object exposing build / search.
    Use this: as the sparse (keyword) half of hybrid retrieval — catches
    exact terms (codes, names, rare words) that dense embeddings can blur
    together with semantically similar but literally different text.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[Document] = []

    def build(self, documents: list[Document]) -> None:
        """
        Takes: the full set of Documents to index (same corpus dense search
        uses).
        Returns: nothing — indexes in memory for search() to use.
        Use this: once per process (or whenever the corpus changes), before
        calling search().
        """
        self._documents = documents
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, k: int = settings.retrieval_top_k) -> list[tuple[Document, float]]:
        """
        Takes: a query string and how many results to return.
        Returns: up to k (Document, score) pairs ranked by BM25 score,
        highest first. Score is an unbounded positive number (not a
        probability or a 0-1 range) — RRF only needs the rank order, so the
        raw scale never has to be reconciled with dense search's scale.
        Use this: as the sparse retriever input to hybrid_fusion.

        Returns an empty list if build() hasn't been called or the corpus
        is empty — an empty index is a valid state, not a caller error.
        """
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._documents, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    # Tiny self-test: a query containing a rare literal keyword should rank
    # the document with that exact keyword at the top.
    sample_docs = [
        Document(page_content="RRF fuses BM25 and dense retrieval by summing reciprocal ranks.", metadata={"source": "rrf.txt"}),
        Document(page_content="Tesseract is an open-source OCR engine for reading text out of images.", metadata={"source": "ocr.txt"}),
        Document(page_content="Order SKU-4471 was flagged for a billing discrepancy last quarter.", metadata={"source": "billing.txt"}),
    ]
    retriever = BM25Retriever()
    retriever.build(sample_docs)

    query = "What happened with SKU-4471?"
    results = retriever.search(query, k=2)
    print(f"Top {len(results)} result(s) for {query!r}:")
    for doc, score in results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r} text={doc.page_content!r}")
