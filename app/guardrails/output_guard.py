"""
Catch PII leakage in the LLM's generated answer before it reaches the user.

Same NeMo Guardrails mechanism as input_guard.py (RailsConfig + LLMRails,
NeMo's built-in `regex` library rail, `check_async(..., rail_types=[OUTPUT])`
— no LLM self-check call needed), applied to the *output* side of a turn
instead of the input side. See input_guard.py's module docstring for the
full reasoning on why deterministic regex over an LLM self-check rail.

Why this works: reuses
input_guard.PII_PATTERNS rather than redefining an equivalent list here —
PII has the same shape whether it shows up in a user's question or a
model's answer, and keeping one definition means a future fix to, say, the
phone-number pattern doesn't have to be remembered in two places.

What this catches, and what it deliberately doesn't:

Catches: the model's answer echoing back something PII-shaped — most
realistically, a retrieved document containing a customer's email or phone
number that the LLM repeats verbatim in its answer. This is the actual risk
surface for a RAG system specifically (the model isn't inventing PII from
nothing; it's relaying PII that was already sitting in a retrieved chunk).

Deliberately not attempting: general jailbreak/unsafe-content detection on
output. Two reasons, not just "out of scope": (1) this system's generation
prompt (app/graph/nodes.py generate_node) already constrains the model to
answer only from retrieved context, which meaningfully narrows the output
surface compared to an open-ended chatbot; (2) a real "did the model comply
with a jailbreak" check needs semantic judgment (an LLM-as-judge or a
classifier model), which is exactly the LLM-call-on-every-request tradeoff
this design chose to avoid for the *blocking* decision. PII leakage is the
one output risk that's actually shape-matchable with regex — that's why
it's the one this file handles.

Two natural follow-up questions:
1. "Why not also block output that fails a groundedness check?" —
   groundedness is a spectrum judgment (how much of the answer is
   supported by context), not a binary structural match; that's what
   app/evaluation/eval_metrics.py's context_overlap_ratio is for, as an
   offline eval metric, not a runtime pass/fail block. Blocking every
   partially-ungrounded answer at runtime would need a real faithfulness
   model, not regex.
2. "What if the retrieved context itself contains PII the answer needs to
   legitimately reference?" — this rail would block that answer too; it
   has no way to distinguish "leaking someone else's PII" from "correctly
   answering a question that happens to involve a phone number." That's a
   real false-positive cost of a purely structural check, traded
   deliberately for zero LLM calls and full determinism.
"""

import asyncio

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from app.config.settings import settings
from app.guardrails.input_guard import PII_PATTERNS, GuardResult

_COLANG_CONTENT = """
define bot refuse to respond
  "I can't share that response as-is."
"""

_rails: LLMRails | None = None


def _get_rails() -> LLMRails:
    """Lazily build the output-rail engine once per process, not once per call."""
    global _rails
    if _rails is None:
        pattern_lines = "\n".join(f"            - '{p}'" for p in PII_PATTERNS)
        yaml_content = f"""
models:
  - type: main
    engine: ollama
    model: {settings.ollama_model}

rails:
  output:
    flows:
      - regex check output
  config:
    regex_detection:
      output:
        patterns:
{pattern_lines}
        case_insensitive: true
"""
        config = RailsConfig.from_content(colang_content=_COLANG_CONTENT, yaml_content=yaml_content)
        _rails = LLMRails(config)
    return _rails


def check_output(query: str, answer: str) -> GuardResult:
    """
    Takes: the user's query (for rail context) and the model's generated
    answer.
    Returns: a GuardResult over the *answer* — allowed=False means this
    answer should not be shown to the user as-is.
    Use this: as the last check on a generated answer, right before it's
    returned to the caller.
    """
    rails = _get_rails()
    result = asyncio.run(
        rails.check_async(
            [{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
            rail_types=[RailType.OUTPUT],
        )
    )
    if result.status != RailStatus.BLOCKED:
        return GuardResult(allowed=True, reason=None, text=answer)
    return GuardResult(allowed=False, reason="pii_detected", text=answer)


if __name__ == "__main__":
    # Tiny self-test: an answer that leaks an email should be blocked; a
    # clean answer should pass through.
    leaky_result = check_output(
        "What's the support contact?", "You can reach support at help@example.com."
    )
    print(f"Leaky answer:  allowed={leaky_result.allowed} reason={leaky_result.reason!r}")
    assert not leaky_result.allowed and leaky_result.reason == "pii_detected"

    clean_result = check_output(
        "What is reciprocal rank fusion?",
        "RRF merges dense and BM25 rankings by summing reciprocal ranks.",
    )
    print(f"Clean answer:  allowed={clean_result.allowed} reason={clean_result.reason!r}")
    assert clean_result.allowed

    print("\nOK: PII-leaking answer blocked, clean answer passed.")
