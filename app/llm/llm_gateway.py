"""
Single entry point for calling an LLM to get a text completion.

This is deliberately minimal — just enough for the query-rewrite and
query-expansion modules (Day 3) to have a real model to call, ahead of its
nominal Day 6 slot in the build plan. Everything downstream (chat, RAG
answer generation) is still Day 6 scope and will extend this file, not
replace it.

Manual provider switch, not an abstraction library: CLAUDE.md rules out
LiteLLM, so provider selection is a plain if/else on settings.llm_provider.
Groq is the default (free-tier, fast); Ollama is the local fallback for
when there's no API key or no network at all.
"""

from groq import Groq

from app.config.settings import settings


def get_completion(prompt: str, system: str | None = None) -> str:
    """
    Takes: a user prompt, and an optional system prompt for instructions
    that should shape the whole response rather than being part of the
    input text.
    Returns: the model's text response, stripped of surrounding whitespace.
    Use this: anywhere a module needs a single LLM call — query rewrite,
    query expansion, and eventually the chat endpoint's answer generation.

    Fails loudly if the configured provider is Groq but no API key is set —
    silently falling back to Ollama here would hide a missing .env value
    behind a confusing "wrong model answered" symptom instead of a clear
    configuration error.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "LLM_PROVIDER is 'groq' but GROQ_API_KEY is empty. "
                "Set it in .env, or switch LLM_PROVIDER to 'ollama'."
            )
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    # Fallback: a local Ollama server, no API key needed.
    import ollama

    response = ollama.chat(model=settings.ollama_model, messages=messages)
    return response["message"]["content"].strip()


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
