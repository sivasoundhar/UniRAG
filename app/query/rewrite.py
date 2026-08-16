"""
Rewrite a raw user query into a cleaner one before it hits retrieval.

Why this prompt is shaped the way it is: retrieval (BM25 especially, but
dense too) rewards queries that look like the text being searched for —
specific nouns, no filler. Users don't type that way; they type things like
"whats that thing about the rank fusion again" mid-conversation. The prompt
below asks for three specific transformations (typos, filler, pronoun/
reference resolution) rather than an open-ended "improve this query,"
because an open-ended instruction invites the model to also answer the
question, add commentary, or over-elaborate a short query into a long one —
all of which make retrieval worse, not better. Constraining the output to
"return only the rewritten query, nothing else" is doing real work here: it's
the difference between a drop-in replacement string and a paragraph that
needs its own parsing step.
"""

from app.llm.llm_gateway import get_completion

_SYSTEM_PROMPT = (
    "You rewrite search queries for a retrieval system. Given a user's "
    "query, fix typos, remove filler words, and resolve vague references "
    "(e.g. 'that thing', 'it') into specific terms if the query provides "
    "enough context to do so. Keep the original intent and meaning exactly. "
    "Return only the rewritten query on a single line — no explanation, "
    "no quotation marks, no commentary."
)


def rewrite_query(query: str) -> str:
    """
    Takes: a raw user query, possibly sloppy or ambiguous.
    Returns: a single rewritten query string, retrieval-friendly.
    Use this: right before calling retrieval, on every user turn — cheap
    enough to always run rather than trying to detect when it's "needed."
    """
    return get_completion(query, system=_SYSTEM_PROMPT)


if __name__ == "__main__":
    # Tiny self-test: a sloppy, filler-heavy query in, a clean one out.
    from app.config.settings import settings

    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env — skipping live call.")
        print("Set it and re-run to see a real rewrite.")
    else:
        original = "umm so like whats that rank thing that combines bm25 and the vector search again"
        rewritten = rewrite_query(original)
        print(f"Original:  {original!r}")
        print(f"Rewritten: {rewritten!r}")
