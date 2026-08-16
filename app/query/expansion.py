"""
Expand one query into several phrasings to widen retrieval recall.

Why this prompt is shaped the way it is: rewrite.py fixes one query into
one better query; this module is for the opposite failure mode — a single
well-formed query that still only overlaps part of the vocabulary a
relevant chunk might use (e.g. "LLM" vs "language model" vs "chatbot").
Asking for exactly settings.query_expansion_n variants, one per line, no
numbering, is a parsing decision as much as a prompt one: a numbered list
("1. ...") or bullet format needs regex to strip formatting before the
variants are usable as search queries, so the prompt asks for the plain
strings up front instead of cleaning up after the fact. Each variant is
told to preserve intent, not just add unrelated keywords — expansion that
drifts off-topic increases retrieval noise instead of recall.
"""

from app.config.settings import settings
from app.llm.llm_gateway import get_completion

_SYSTEM_PROMPT_TEMPLATE = (
    "You expand a search query into {n} alternative phrasings that preserve "
    "the original intent but use different wording, synonyms, or related "
    "terms a relevant document might use instead. Return exactly {n} lines, "
    "one phrasing per line, no numbering, no bullets, no explanation."
)


def expand_query(query: str, n: int = settings.query_expansion_n) -> list[str]:
    """
    Takes: a query string and how many alternative phrasings to generate.
    Returns: a list of up to n alternative query strings (fewer if the
    model returns fewer non-empty lines than asked for).
    Use this: alongside the original query when retrieving — run each
    variant through retrieval and merge results (e.g. via hybrid_fusion)
    to catch relevant chunks that use different vocabulary than the query.
    """
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(n=n)
    response = get_completion(query, system=system_prompt)
    variants = [line.strip() for line in response.splitlines() if line.strip()]
    return variants[:n]


if __name__ == "__main__":
    # Tiny self-test: one query in, several differently-worded variants out.
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env — skipping live call.")
        print("Set it and re-run to see real expansions.")
    else:
        original = "how does hybrid search combine keyword and semantic ranking?"
        variants = expand_query(original)
        print(f"Original: {original!r}")
        print(f"Expanded into {len(variants)} variant(s):")
        for variant in variants:
            print(f"  {variant!r}")
