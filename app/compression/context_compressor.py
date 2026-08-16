"""
Trim each retrieved chunk down to its most query-relevant sentences.

By the time context reaches here, retrieval + fusion + rerank have already
narrowed the whole corpus down to a handful of chunks (settings.rerank_top_n).
This step narrows further, within each chunk: a 500-character chunk
(settings.chunk_size) usually has one or two sentences that actually answer
the query and several more that are just surrounding context. Passing the
whole chunk to the LLM spends tokens on the surrounding sentences too; this
keeps only the sentences worth paying for.

No LLM call here — reusing BGEEmbedder (already loaded for retrieval) to
score sentences against the query keeps this step free and fast, unlike an
LLM-based extractive-compression approach that would cost a model call per
chunk.
"""

import re

import numpy as np
from langchain_core.documents import Document

from app.config.settings import settings
from app.embeddings.bge_embedder import BGEEmbedder


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def compress_context(
    query: str,
    documents: list[Document],
    sentences_per_doc: int = settings.compression_sentences_per_doc,
) -> list[Document]:
    """
    Takes: the query and a short list of Documents (e.g. rerank output).
    Returns: the same Documents, each with page_content cut down to its
    top sentences_per_doc most query-relevant sentences — re-assembled in
    their original order within the chunk, not similarity order, so the
    compressed text still reads as a coherent excerpt rather than a
    shuffled list of "most relevant" fragments.
    Use this: as the last step before building the prompt for generation.

    A document with sentences_per_doc or fewer sentences already is
    returned unchanged — there's nothing to cut.
    """
    embedder = BGEEmbedder()
    query_vec = np.array(embedder.embed_query(query))

    compressed: list[Document] = []
    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if len(sentences) <= sentences_per_doc:
            compressed.append(doc)
            continue

        sentence_vecs = np.array(embedder.embed_documents(sentences))
        similarities = sentence_vecs @ query_vec  # normalized vectors, so dot product == cosine similarity

        top_indices = sorted(np.argsort(similarities)[-sentences_per_doc:])
        kept_text = " ".join(sentences[i] for i in top_indices)
        compressed.append(Document(page_content=kept_text, metadata=dict(doc.metadata)))

    return compressed


if __name__ == "__main__":
    # Tiny self-test: a chunk with one on-topic sentence buried among
    # off-topic ones about an unrelated subject — compression should keep
    # the on-topic sentence(s) and drop the unrelated one.
    text = (
        "RRF fuses BM25 and dense retrieval by summing reciprocal ranks. "
        "Reciprocal rank fusion does not require normalising scores onto a shared scale. "
        "Tesseract is a completely unrelated OCR engine for reading scanned images. "
        "The smoothing constant k controls how much rank 1 dominates the fused score."
    )
    doc = Document(page_content=text, metadata={"source": "rrf.txt"})

    query = "How does reciprocal rank fusion work?"
    result = compress_context(query, [doc], sentences_per_doc=2)[0]

    print(f"Original ({len(_split_sentences(text))} sentences): {text!r}")
    print(f"Compressed (2 sentences): {result.page_content!r}")
    assert "Tesseract" not in result.page_content, "off-topic OCR sentence should have been dropped"
    print("OK: off-topic sentence dropped, on-topic sentences kept.")
