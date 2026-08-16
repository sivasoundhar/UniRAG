"""
Narrow retrieval to documents matching metadata (e.g. source file, doc type).

Two entry points because dense and sparse retrieval filter differently.
Chroma can filter natively at query time via a `where` clause — pushing the
filter down means it never has to embed-compare documents that couldn't
match anyway. rank_bm25 has no such hook (BM25Okapi just scores whatever
token lists it was built with), so BM25/RRF results need the equivalent
filter applied afterward, on the Documents' own metadata.
"""

from langchain_core.documents import Document


def build_where_clause(filters: dict[str, str | int]) -> dict:
    """
    Takes: a flat dict of metadata field -> required value, e.g.
    {"source": "handbook.pdf"}.
    Returns: a Chroma-compatible `where` clause. A single filter becomes a
    single `$eq` comparison; multiple filters are combined with `$and`,
    since Chroma requires an explicit boolean operator once there's more
    than one condition rather than accepting a flat multi-key dict.
    Use this: pass straight to VectorStore/Chroma's query(where=...) to
    filter dense search before it runs, not after.

    Takes: nothing to filter on -> returns an empty dict, meaning "no
    filter," rather than raising — an unfiltered search is a normal request.
    """
    if not filters:
        return {}

    conditions = [{field: {"$eq": value}} for field, value in filters.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def filter_documents(
    results: list[tuple[Document, float]], filters: dict[str, str | int]
) -> list[tuple[Document, float]]:
    """
    Takes: a ranked list of (Document, score) pairs (e.g. BM25 or RRF
    output) and the same flat metadata filter dict as build_where_clause.
    Returns: the subset of pairs whose Document.metadata matches every
    filter key/value exactly, in their original relative order.
    Use this: as the post-hoc equivalent of build_where_clause for
    retrievers (BM25, fused RRF results) that can't filter natively.
    """
    if not filters:
        return results

    return [
        (doc, score)
        for doc, score in results
        if all(doc.metadata.get(field) == value for field, value in filters.items())
    ]


if __name__ == "__main__":
    # Tiny self-test: a mixed-source corpus narrowed down to one source.
    sample_results = [
        (Document(page_content="RRF fuses BM25 and dense retrieval.", metadata={"source": "rrf.txt"}), 0.9),
        (Document(page_content="Tesseract reads text out of scanned images.", metadata={"source": "ocr.txt"}), 0.8),
        (Document(page_content="ChromaDB persists embeddings to disk.", metadata={"source": "rrf.txt"}), 0.7),
    ]

    where_clause = build_where_clause({"source": "rrf.txt"})
    print(f"Chroma where clause: {where_clause}")

    filtered = filter_documents(sample_results, {"source": "rrf.txt"})
    print(f"Filtered {len(sample_results)} -> {len(filtered)} result(s):")
    for doc, score in filtered:
        print(f"  source={doc.metadata['source']!r} text={doc.page_content!r}")
