"""
Single entry point for calling an LLM to get a text completion, and (new)
for asking a vision-capable model to describe an image.

This is deliberately minimal — just enough for the query-rewrite and
query-expansion modules to have a real model to call. Everything
downstream (chat, RAG answer generation, and now image understanding)
extends this file, not replaces it.

Manual provider *selection*, not an abstraction library: no LiteLLM or
similar dependency, so which provider a fresh request starts on is still a plain
if/else on settings.llm_provider. What's automatic is the fallback once a
request is underway: settings.llm_provider="groq" tries Groq (with its own
model-level retry, see _call_groq) and only drops to Ollama if every Groq
option is actually exhausted — not decided ahead of time by a human
flipping a setting. Originally this file left even that manual, but real
use showed the actual failure mode is a connection-level fault that
recurs for the whole session (a stuck network condition, not a one-off) —
something a human had to notice and fix by restarting with a different
LLM_PROVIDER every single time, which is precisely the toil automatic
fallback exists to remove.

Vision (get_vision_completion) is Groq-only, no Ollama fallback — unlike
the text path, this project isn't running a local vision model (not
installed, and irrelevant for a hosted demo deployment that won't have
Ollama available at all). See settings.groq_vision_model's comment for
why those specific model IDs and where to fix them if wrong.
"""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from groq import APIStatusError, Groq, GroqError

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Completion:
    """
    Takes: nothing directly — built by _call_groq/_call_ollama/get_*.
    Returns: n/a.
    Use this: whenever a caller needs to know *which* model actually
    answered, not just the text — e.g. the chat endpoint showing "answered
    by llama-3.1-8b-instant" instead of asserting a fixed model name that
    might not be the one that actually ran, given automatic fallback can
    silently swap in a different model or provider mid-request.
    """

    text: str
    model: str
    provider: str  # "groq" or "ollama"


def _call_groq(messages: list[dict], models_to_try: list[str], **extra_params) -> Completion:
    """
    Takes: chat messages already in Groq's format, the ordered list of
    model IDs to try (settings.groq_model + its fallbacks for text calls,
    settings.groq_vision_model + its fallbacks for vision calls — same
    retry mechanics either way, only the model list differs), and any
    extra keyword args to pass straight through to
    client.chat.completions.create (e.g. reasoning_effort, max_completion_
    tokens — needed for the vision model specifically, see
    get_vision_completion, but not the plain text models, which is why
    this isn't hard-coded here).
    Returns: the first successful completion, tagged with which model
    actually produced it.
    Use this: from get_completion()/get_vision_completion(), when
    settings.llm_provider is "groq" — kept separate so the retry loop
    doesn't clutter the provider-selection logic callers actually care
    about.

    Raises (doesn't itself fall back to Ollama) if every model in the list
    fails — that's the caller's job, one layer up, since it's the one that
    knows whether an Ollama fallback even applies (text: yes; vision: no,
    see get_vision_completion).
    """
    client = Groq(api_key=settings.groq_api_key)
    last_error: Exception | None = None

    for model in models_to_try:
        try:
            response = client.chat.completions.create(model=model, messages=messages, **extra_params)
            return Completion(text=response.choices[0].message.content.strip(), model=model, provider="groq")
        except APIStatusError as e:
            # APIStatusError means Groq actually answered (rate limit,
            # decommissioned/unknown model, that model's own server-side
            # fault) — a different model in the list has a real chance of
            # succeeding, so retry. Deliberately NOT catching the broader
            # GroqError here: its other subclass, APIConnectionError, means
            # no response came back at all (e.g. a network-level TLS
            # interception issue) — that fails identically for every
            # model since the fault is the connection, not the model, so it
            # propagates immediately instead of burning time on more
            # doomed attempts.
            logger.warning("Groq model %r failed (%s), trying next", model, e)
            last_error = e

    raise last_error  # every model in the list failed


def _call_ollama(messages: list[dict]) -> Completion:
    """
    Takes: chat messages already in Ollama's format (same shape as Groq's —
    {"role": ..., "content": ...} — so callers don't need to know which
    provider ends up handling a given call).
    Returns: the completion, tagged with settings.ollama_model.
    Use this: from get_completion(), either because settings.llm_provider
    is "ollama" outright, or because Groq was tried first and exhausted
    every option.
    """
    import ollama

    response = ollama.chat(model=settings.ollama_model, messages=messages)
    return Completion(text=response["message"]["content"].strip(), model=settings.ollama_model, provider="ollama")


def _complete(messages: list[dict]) -> Completion:
    """
    Takes: chat messages already in provider format.
    Returns: the completion, from whichever provider/model actually
    answered.
    Use this: the shared provider-selection + fallback logic behind
    get_completion() and get_completion_with_model() — both need the exact
    same behavior, just different return shapes for backward compatibility
    (see get_completion's docstring).

    Fails loudly if the configured provider is Groq but no API key is set —
    silently falling back to Ollama here would hide a missing .env value
    behind a confusing "wrong model answered" symptom instead of a clear
    configuration error, and it's a one-time fix (add the key), not a
    transient condition worth retrying around.

    Once a key IS set, though, a genuine Groq-side failure — every one of
    _call_groq's models rejected the request, or the connection itself is
    down (a network-level TLS interception issue, in practice) — falls
    back to Ollama automatically instead of raising.
    Unlike the missing-key case, this isn't a config typo to fix once; it's
    proven to be a condition that persists for a whole session and recurs
    across many, which previously meant a human manually restarting the
    process with LLM_PROVIDER=ollama every single time it happened. Logged
    as a warning either way, so "Groq is actually down right now" stays
    visible in the logs even though the request itself still succeeds.
    """
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is empty. "
                "Set it in .env, or switch LLM_PROVIDER to 'ollama'."
            )
        try:
            return _call_groq(messages, [settings.groq_model, *settings.groq_fallback_models])
        except GroqError as e:
            logger.warning(
                "Groq failed across every fallback model (%s) — falling back to Ollama for this request", e
            )
            return _call_ollama(messages)

    return _call_ollama(messages)


def get_completion(prompt: str, system: str | None = None) -> str:
    """
    Takes: a user prompt, and an optional system prompt for instructions
    that should shape the whole response rather than being part of the
    input text.
    Returns: the model's text response, stripped of surrounding whitespace.
    Use this: anywhere a module needs a single LLM call and doesn't care
    which model answered — query rewrite, query expansion. Unchanged
    signature/return type from before get_completion_with_model existed,
    so those callers needed zero changes when this file grew that.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _complete(messages).text


def get_completion_with_model(prompt: str, system: str | None = None) -> Completion:
    """
    Takes: same as get_completion.
    Returns: the full Completion (text + which model/provider answered),
    not just the text.
    Use this: from app/graph/nodes.py's generate_node — the chat endpoint
    is where a user actually benefits from seeing which model answered,
    given automatic Groq-model and Groq->Ollama fallback both mean the
    answering model isn't fixed/predictable from settings alone anymore.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _complete(messages)


def get_vision_completion(image_path: str, prompt: str) -> Completion:
    """
    Takes: a path to an image file, and a text prompt asking about it
    (e.g. "Describe what this image shows in detail.").
    Returns: the model's text description, tagged with which vision model
    produced it.
    Use this: from app/loaders/ocr_loader.py, when OCR extracts no text
    from an uploaded image — a photo/scan with nothing printed on it (an
    X-ray, a diagram, a CAE render) needs a model that can actually look
    at the pixels, which plain OCR fundamentally cannot do.

    Groq-only, no Ollama fallback, unlike get_completion — see this
    module's docstring for why. Raises if both
    settings.groq_vision_model and its fallbacks fail; there's nothing
    left to try after that.

    reasoning_effort="none" is load-bearing, not a nice-to-have: verified
    live that settings.groq_vision_model (a "thinking" model) otherwise
    dumps its entire chain-of-thought into the response — for one real
    test image, over 500 words of second-guessing itself about which
    finger an arrow pointed to, then got cut off mid-sentence by
    max_completion_tokens before ever stating an actual answer. Useless as
    an indexed chunk either way (all reasoning noise, or truncated with no
    conclusion at all). With reasoning_effort="none", the same image
    produced a clean 3-sentence factual description on the first try.
    max_completion_tokens is still capped, as a safety net in case a
    future model swapped in via settings ignores reasoning_effort.
    """
    if not settings.groq_api_key:
        raise ValueError(
            "Vision calls require GROQ_API_KEY — no local vision model is "
            "configured as a fallback. Set GROQ_API_KEY in .env."
        )

    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Groq's vision input is OpenAI-compatible: an image_url content part
    # whose "url" can be a data: URI instead of an actual hosted URL, so a
    # local file never needs to be uploaded anywhere else first.
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }
    ]
    return _call_groq(
        messages,
        [settings.groq_vision_model, *settings.groq_vision_fallback_models],
        reasoning_effort="none",
        max_completion_tokens=600,
    )


if __name__ == "__main__":
    # Tiny self-test: skip the actual call (and print why) if there's no
    # key configured yet, rather than crashing the self-test on missing
    # config — this file being runnable doesn't depend on .env being filled in.
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env — skipping live call.")
        print("Set it and re-run to see a real completion.")
    else:
        reply = get_completion("Reply with exactly one word: pong")
        print(f"Model replied: {reply!r}")

        with_model = get_completion_with_model("Reply with exactly one word: pong")
        print(f"With model info: text={with_model.text!r} model={with_model.model!r} provider={with_model.provider!r}")

    # Prove the Groq -> Ollama fallback itself, not just "get_completion
    # returned something" — mocked so this is deterministic regardless of
    # whether Groq happens to be reachable right now, and runs even when
    # settings.llm_provider is "ollama" (the fallback path is still real
    # code worth checking every run, not just when Groq is the active
    # provider).
    from unittest.mock import MagicMock, patch

    import groq

    print("\nVerifying Groq -> Ollama fallback (every Groq model mocked to fail)...")
    # Patch via __name__, not the hardcoded "app.llm.llm_gateway" path:
    # `python -m app.llm.llm_gateway` imports this file twice under two
    # different sys.modules entries (once as "app.llm.llm_gateway" during
    # import machinery, again as "__main__" for this actual running block)
    # — patching the former has zero effect on the get_completion this
    # block calls directly, which resolves Groq/_call_ollama from
    # __main__'s own globals. __name__ is "__main__" right here regardless
    # of which of the two copies is executing, so this always targets the
    # right one.
    with patch(f"{__name__}.Groq") as mock_groq_cls, patch(f"{__name__}._call_ollama") as mock_ollama:
        mock_groq_cls.return_value.chat.completions.create.side_effect = groq.APIConnectionError(
            request=MagicMock()
        )
        mock_ollama.return_value = Completion(text="pong (from Ollama)", model="llama3.2", provider="ollama")

        original_provider = settings.llm_provider
        original_key = settings.groq_api_key
        settings.llm_provider = "groq"
        settings.groq_api_key = "fake-key-for-this-test"
        try:
            result = get_completion("ping")
        finally:
            settings.llm_provider = original_provider
            settings.groq_api_key = original_key

        assert result == "pong (from Ollama)", "get_completion should return Ollama's result after Groq fails"
        assert mock_ollama.called, "Ollama should have been called as the fallback"
    print("OK: every Groq model failing falls back to Ollama automatically, no manual restart needed.")

    # Prove get_vision_completion tries the fallback model too, mocked the
    # same way as above — deterministic regardless of live Groq access,
    # and proves the model-ID-in-a-list mechanism without needing a real
    # image or a working network connection to verify the retry logic.
    # No fallback model to test falling back to — settings.groq_vision_
    # fallback_models is genuinely empty (qwen/qwen3.6-27b is the only
    # vision-capable model this Groq account has, confirmed live against
    # client.models.list()). What's worth proving instead: the call goes
    # to the right (single) model, and — the actually load-bearing part —
    # reasoning_effort="none" and max_completion_tokens are really sent,
    # not just documented. Without reasoning_effort="none", this same
    # model was verified to dump 500+ words of unfinished chain-of-thought
    # instead of an answer (see get_vision_completion's docstring).
    print("\nVerifying vision call sends the right model and reasoning params...")
    with patch(f"{__name__}.Groq") as mock_groq_cls, patch(f"{__name__}.Path") as mock_path_cls:
        mock_path_cls.return_value.read_bytes.return_value = b"fake-image-bytes"
        reply = MagicMock()
        reply.choices = [MagicMock(message=MagicMock(content=" a hand X-ray, four panels, arrow at 5th metacarpal "))]
        mock_groq_cls.return_value.chat.completions.create.return_value = reply

        original_key = settings.groq_api_key
        settings.groq_api_key = "fake-key-for-this-test"
        try:
            result = get_vision_completion("fake.jpg", "Describe this image.")
        finally:
            settings.groq_api_key = original_key

        call_kwargs = mock_groq_cls.return_value.chat.completions.create.call_args.kwargs
        assert result.text == "a hand X-ray, four panels, arrow at 5th metacarpal"
        assert result.model == settings.groq_vision_model
        assert call_kwargs["reasoning_effort"] == "none", "must disable thinking mode, or responses are unusable"
        assert call_kwargs["max_completion_tokens"] == 600
    print("OK: vision call targets the right model with reasoning disabled.")
