"""
Per-conversation turn history, so a multi-turn chat can refer back to itself.

Storage is a plain in-memory dict, keyed by conversation_id — it does not
survive a process restart. That's a deliberate stub, not an oversight: this
is the Day 4 finish line's memory requirement (one conversation carries
context across turns within a single run), not the long-term/persistent
memory the tech stack table explicitly marks as a stub-with-a-TODO.

TODO(long-term memory): swap the module-level dict below for a real store
(Redis, Postgres) keyed the same way by conversation_id, once conversations
need to survive a restart or be shared across API workers. Every function
signature here should stay the same — only the storage backing changes.
"""

from app.config.settings import settings

_conversations: dict[str, list[dict[str, str]]] = {}


def append_turn(conversation_id: str, role: str, content: str) -> None:
    """
    Takes: the conversation to append to, who said it ("user" or
    "assistant"), and what was said.
    Returns: nothing — appends in place.
    Use this: once per message, right after a user query arrives and again
    right after the model answers it.
    """
    _conversations.setdefault(conversation_id, []).append({"role": role, "content": content})


def get_history(
    conversation_id: str, max_turns: int = settings.conversation_memory_max_turns
) -> list[dict[str, str]]:
    """
    Takes: a conversation_id and how many turn-*pairs* (user+assistant) to
    return at most.
    Returns: the most recent min(max_turns, available) turn-pairs, oldest
    first, as a flat list of {"role", "content"} messages — i.e. up to
    2 * max_turns messages, not max_turns messages.
    Use this: to build the chat-history portion of a prompt (rewrite,
    generate) without replaying an entire long conversation every turn.

    Returns an empty list for an unknown conversation_id rather than
    raising — a brand new conversation has no history, which is a normal
    starting state, not an error.
    """
    messages = _conversations.get(conversation_id, [])
    return messages[-(max_turns * 2):]


def clear(conversation_id: str) -> None:
    """
    Takes: a conversation_id.
    Returns: nothing — removes all stored turns for that conversation, if any.
    Use this: when a user starts a new chat and the old one should stop
    influencing rewrite/generate.
    """
    _conversations.pop(conversation_id, None)


if __name__ == "__main__":
    # Tiny self-test: append more turn-pairs than max_turns allows, confirm
    # get_history trims to the most recent ones and keeps order intact.
    conversation_id = "demo"
    for i in range(1, 4):
        append_turn(conversation_id, "user", f"question {i}")
        append_turn(conversation_id, "assistant", f"answer {i}")

    history = get_history(conversation_id, max_turns=2)
    print(f"History (max_turns=2): {history}")
    assert len(history) == 4, "expected 2 turn-pairs = 4 messages"
    assert history[0]["content"] == "question 2", "oldest kept turn should be question 2, not question 1"

    clear(conversation_id)
    assert get_history(conversation_id) == []
    print("OK: max_turns trims correctly, and clear() empties the conversation.")
