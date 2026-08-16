"""
Smoke tests for app/retrieval and app/rerank — real assert-based pytest
versions of each module's own __main__ self-test, on a tiny shared corpus.
"""

from langchain_core.documents import Document

from app.rerank.cross_encoder_rerank import rerank
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion
from app.retrieval.vector_store import VectorStore


def _sample_docs() -> list[Document]:
    # Three documents, not two: with only two docs, a term appearing in
    # exactly one of them gets BM25 IDF = log((2-1+0.5)/(1+0.5)) = log(1) = 0
    # — every query contributes nothing and results fall back to insertion
    # order. A third distractor gives the IDF math something real to do.
    return [
        Document(
            page_content="RRF fuses BM25 and dense retrieval by summing reciprocal ranks.",
            metadata={"source": "rrf.txt"},
        ),
        Document(
            page_content="Tesseract is an open-source OCR engine for reading text out of images.",
            metadata={"source": "ocr.txt"},
        ),
        Document(
            page_content="ChromaDB persists embeddings to disk so the index survives a restart.",
            metadata={"source": "chroma.txt"},
        ),
    ]


def test_vector_store_add_and_search():
    store = VectorStore()
    store.add_documents(_sample_docs())

    results = store.similarity_search("How does reciprocal rank fusion work?", k=2)

    assert len(results) == 2
    assert any("RRF" in doc.page_content for doc, _score in results)


def test_bm25_retriever_finds_keyword_match():
    retriever = BM25Retriever()
    retriever.build(_sample_docs())

    results = retriever.search("Tesseract OCR", k=2)

    assert results[0][0].metadata["source"] == "ocr.txt"


def test_reciprocal_rank_fusion_merges_ranked_lists():
    doc_a = Document(page_content="a", metadata={"chunk_id": "a"})
    doc_b = Document(page_content="b", metadata={"chunk_id": "b"})

    fused = reciprocal_rank_fusion([[(doc_a, 0.9), (doc_b, 0.5)], [(doc_b, 3.0), (doc_a, 1.0)]])

    assert {doc.metadata["chunk_id"] for doc, _score in fused} == {"a", "b"}


def test_rerank_reorders_by_relevance():
    query = "How does reciprocal rank fusion work?"
    candidates = [
        Document(page_content="Tesseract is an open-source OCR engine.", metadata={"source": "ocr.txt"}),
        Document(
            page_content="RRF fuses BM25 and dense retrieval by summing reciprocal ranks.",
            metadata={"source": "rrf.txt"},
        ),
    ]

    results = rerank(query, candidates, top_n=2)

    assert results[0][0].metadata["source"] == "rrf.txt"
