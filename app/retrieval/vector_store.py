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

# Shared by every read path that should skip "temporarily deleted" chunks
# (similarity_search, get_all_documents's default). $ne True/False rather
# than $eq True: verified empirically that Chroma's $ne matches a document
# missing the field entirely, so this stays correct for chunks indexed
# before the "active" key existed, without a data migration.
_ACTIVE_FILTER = {"active": {"$ne": False}}


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
            # Explicit rather than relying on the key's absence to mean
            # "active" — set_active_by_source below toggles this same key
            # to False for a soft ("temporary") delete, so every chunk
            # should carry it from the start rather than only the ones
            # that have ever been hidden.
            metadata["active"] = True
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
            # $ne (not $eq True) so chunks indexed before the "active" flag
            # existed — which have no "active" key at all — still count as
            # active instead of silently vanishing from retrieval; only an
            # explicit active=False (a "temporary" delete) is excluded.
            where=_ACTIVE_FILTER,
        )

        pairs: list[tuple[Document, float]] = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = 1 - distance
            pairs.append((Document(page_content=text, metadata=metadata), score))
        return pairs

    def delete_by_source(self, source: str) -> int:
        """
        Takes: the exact metadata["source"] value to remove — the original
        filename set at upload time (see app/main.py's upload route).
        Returns: how many chunks were deleted (0 if nothing matched).
        Use this: to remove a document from the corpus by name, without the
        caller needing to know individual chunk_ids — e.g. from a DELETE
        API route cleaning up an unwanted upload.

        Fetches matching ids first (include=[] — ids come back regardless,
        so there's no need to also pull documents/embeddings/metadatas just
        to count them) rather than trusting collection.delete()'s return
        value, because Chroma's delete doesn't report how many rows matched.
        """
        matches = self._collection.get(where={"source": source}, include=[])
        ids = matches["ids"]
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def get_all_documents(self, include_inactive: bool = False) -> list[Document]:
        """
        Takes: include_inactive — pass True to also return chunks that were
        "temporarily deleted" (set_active_by_source(..., False)).
        Returns: every active chunk currently in the collection by default,
        as Documents with their chunk_id intact in metadata.
        Use this: to build the BM25 index over the exact same corpus Chroma
        holds, instead of maintaining a second copy of the documents — the
        default (active only) keeps a hidden document out of BM25 too, not
        just dense search. Pass include_inactive=True only for something
        that needs to see hidden documents, e.g. listing them to restore.
        """
        if self._collection.count() == 0:
            return []

        where = None if include_inactive else _ACTIVE_FILTER
        results = self._collection.get(where=where, include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(results["documents"], results["metadatas"])
        ]

    def set_active_by_source(self, source: str, active: bool) -> int:
        """
        Takes: the exact metadata["source"] value, and the active state to
        set it to (False = "temporary delete" — hidden from retrieval but
        embeddings kept, so restoring is instant with no re-upload/re-embed;
        True = restore).
        Returns: how many chunks were updated (0 if nothing matched).
        Use this: for the "temporary delete" / restore half of document
        management — delete_by_source is the "permanent" half.

        Matches by source with no active-state filter (unlike the read
        paths above) — restoring a hidden document has to be able to find
        it precisely because it's hidden, and re-hiding an already-hidden
        one should be a harmless no-op, not a "not found."

        collection.update() merges the given metadata keys into each
        existing dict rather than replacing it wholesale (verified against
        Chroma directly before relying on it here) — so this can pass just
        {"active": active} without first fetching and re-sending source/
        chunk_id/every other metadata key, and without risking wiping them.
        """
        matches = self._collection.get(where={"source": source}, include=[])
        ids = matches["ids"]
        if not ids:
            return 0
        self._collection.update(ids=ids, metadatas=[{"active": active}] * len(ids))
        return len(ids)


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

    # Temporary delete: hide ocr.txt, confirm it drops out of both dense
    # search and the default (active-only) get_all_documents, but is still
    # in the collection (include_inactive=True) and comes back on restore.
    hidden = store.set_active_by_source("ocr.txt", active=False)
    active_sources = {doc.metadata["source"] for doc in store.get_all_documents()}
    all_sources = {doc.metadata["source"] for doc in store.get_all_documents(include_inactive=True)}
    dense_sources = {doc.metadata["source"] for doc, _ in store.similarity_search("OCR engine", k=5)}
    print(f"\nHid {hidden} chunk(s) with source='ocr.txt'")
    assert hidden == 1 and "ocr.txt" not in active_sources and "ocr.txt" not in dense_sources
    assert "ocr.txt" in all_sources  # still there, just not active
    print("OK: temporary delete hides from get_all_documents and similarity_search, but keeps the embedding.")

    restored = store.set_active_by_source("ocr.txt", active=True)
    active_sources = {doc.metadata["source"] for doc in store.get_all_documents()}
    assert restored == 1 and "ocr.txt" in active_sources
    print("OK: restore (set_active_by_source(..., True)) brings it back with no re-embedding needed.")

    # Delete one of the just-added docs by source and confirm it's really gone.
    deleted = store.delete_by_source("rrf.txt")
    remaining_sources = {doc.metadata["source"] for doc in store.get_all_documents()}
    print(f"\nDeleted {deleted} chunk(s) with source='rrf.txt'")
    print(f"Remaining sources: {sorted(remaining_sources)}")
    assert deleted == 1 and "rrf.txt" not in remaining_sources
    print("OK: delete_by_source (permanent) removed exactly the targeted document's chunks.")

    # Clean up the rest of this self-test's own data so re-running it (or
    # the real app) doesn't accumulate rrf.txt/ocr.txt/chroma.txt/rerank.txt
    # duplicates in the persistent collection on every run.
    for source in ("ocr.txt", "chroma.txt", "rerank.txt"):
        store.delete_by_source(source)
