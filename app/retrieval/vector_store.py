"""
Persist chunked Documents into Chroma and search them by dense similarity.

This module owns the one Chroma collection UniRAG uses. It's also the
single source of truth for the corpus: app.retrieval.bm25_retriever does not
maintain its own copy of the documents — it gets them from
`VectorStore.get_all_documents()` so dense and sparse retrieval are always
searching over exactly the same chunks, keyed by the same `chunk_id`.
"""

import uuid

import chromadb
from langchain_core.documents import Document

from app.config.settings import settings
from app.embeddings.bge_embedder import BGEEmbedder


class VectorStore:
    """
    Takes: nothing at construction — opens (or creates) the persistent
    Chroma collection named in settings.
    Returns: an object exposing add_documents / similarity_search /
    get_all_documents.
    Use this: as the dense half of hybrid retrieval, and as the corpus of
    record that BM25 indexes from.
    """

    def __init__(self) -> None:
        self._embedder = BGEEmbedder()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Chroma defaults to L2 distance; BGE embeddings are trained for
        # cosine similarity, so the collection is created explicitly in
        # cosine space rather than relying on the default.
        self._collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Takes: a list of chunked Documents (e.g. from a chunker in
        app.chunking).
        Returns: nothing — writes into the persistent Chroma collection.
        Use this: once per chunk, at ingest time.

        Each chunk gets a fresh chunk_id here, stored in metadata, so BM25
        and RRF can refer to "the same chunk" without comparing raw text.
        """
        if not documents:
            return

        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = []
        for doc, chunk_id in zip(documents, ids):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = chunk_id
            metadatas.append(metadata)

        embeddings = self._embedder.embed_documents(texts)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def similarity_search(
        self, query: str, k: int = settings.retrieval_top_k
    ) -> list[tuple[Document, float]]:
        """
        Takes: a query string and how many results to return.
        Returns: up to k (Document, score) pairs ranked by dense similarity,
        highest score first. Score is 1 - cosine_distance, so 1.0 is an
        exact match and it decreases from there — this makes dense scores
        comparable in direction (higher is better) to BM25's, even though
        the two are not on the same scale.
        Use this: as the dense retriever input to hybrid_fusion.

        Returns an empty list rather than raising when the collection has
        no documents yet — an empty index is a valid (if uninteresting)
        state, not a caller error.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
        )

        pairs: list[tuple[Document, float]] = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = 1 - distance
            pairs.append((Document(page_content=text, metadata=metadata), score))
        return pairs

    def get_all_documents(self) -> list[Document]:
        """
        Takes: nothing.
        Returns: every chunk currently in the collection, as Documents with
        their chunk_id intact in metadata.
        Use this: to build the BM25 index over the exact same corpus Chroma
        holds, instead of maintaining a second copy of the documents.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(results["documents"], results["metadatas"])
        ]


if __name__ == "__main__":
    # Tiny self-test: index a handful of unrelated sentences, then confirm
    # a query returns the topically matching one first.
    store = VectorStore()
    sample_docs = [
        Document(page_content="RRF fuses BM25 and dense retrieval by summing reciprocal ranks.", metadata={"source": "rrf.txt"}),
        Document(page_content="Tesseract is an open-source OCR engine for reading text out of images.", metadata={"source": "ocr.txt"}),
        Document(page_content="ChromaDB persists embeddings to disk so the index survives a restart.", metadata={"source": "chroma.txt"}),
        Document(page_content="Cross-encoders score a query and passage together for more accurate reranking.", metadata={"source": "rerank.txt"}),
    ]
    store.add_documents(sample_docs)

    query = "How does hybrid search combine keyword and semantic ranking?"
    results = store.similarity_search(query, k=2)
    print(f"Top {len(results)} result(s) for {query!r}:")
    for doc, score in results:
        print(f"  score={score:.4f} source={doc.metadata['source']!r} text={doc.page_content!r}")
