"""
Ephemeral per-session scratch state — a different concern from conversation
history in app/memory/conversation_memory.py.

A conversation's history is "what was said." Session state is "what's true
about this session right now" — e.g. an active metadata filter the user
picked in the UI, which should apply to their next few queries but isn't
part of the chat transcript itself and shouldn't be replayed to the LLM as
if it were a message. Keeping the two separate means clearing one doesn't
accidentally clear the other.

Not wired into the graph (app/graph/build_graph.py): that pipeline is about
conversation memory carrying context across turns. This module exists as
its own concern, ready for the API layer to use — e.g. storing a session's
active filters between requests.
"""

_sessions: dict[str, dict[str, object]] = {}


def set(session_id: str, key: str, value: object) -> None:
    """
    Takes: a session_id, a key, and any value to remember under it.
    Returns: nothing — stores in place, creating the session if new.
    Use this: whenever a session picks up some state that should stick
    around for its next request but isn't a chat message.
    """
    _sessions.setdefault(session_id, {})[key] = value


def get(session_id: str, key: str, default: object = None) -> object:
    """
    Takes: a session_id, a key, and a default to return if either the
    session or the key doesn't exist yet.
    Returns: the stored value, or default.
    Use this: to read back session state without needing to check
    existence first.
    """
    return _sessions.get(session_id, {}).get(key, default)


def clear(session_id: str) -> None:
    """
    Takes: a session_id.
    Returns: nothing — removes all state for that session, if any.
    Use this: when a session ends (e.g. the user logs out or starts fresh).
    """
    _sessions.pop(session_id, None)


if __name__ == "__main__":
    # Tiny self-test: set/get/clear round-trip, plus the default-on-miss case.
    session_id = "demo"
    set(session_id, "active_filter", {"source": "handbook.pdf"})
    print(f"Stored filter: {get(session_id, 'active_filter')}")
    assert get(session_id, "missing_key", "fallback") == "fallback"

    clear(session_id)
    assert get(session_id, "active_filter") is None
    print("OK: set/get round-trips, missing keys return the default, clear() empties the session.")
