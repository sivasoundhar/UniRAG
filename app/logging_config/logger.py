"""
stdlib logging setup, plus one helper that logs the fields every request
needs recorded: latency, retrieval count, sources.

Plain stdlib `logging`, not a structured/JSON logging library — the tech
stack calls for stdlib logging, and a single consistently-shaped message
string is grep-able and readable in a terminal without extra tooling, which
matches this project's "legible over complete" framing.
"""

import logging

from app.config.settings import settings


def get_logger(name: str = "unirag") -> logging.Logger:
    """
    Takes: a logger name (module-qualified names, e.g. __name__, are the
    usual choice).
    Returns: a configured logging.Logger — one console handler, formatted
    with timestamp/level/name, level set from settings.log_level.
    Use this: once per module that needs to log, instead of calling
    logging.getLogger() directly, so every logger in the app shares the
    same format and level without repeating the setup.

    Guards against adding a second handler if called again for the same
    name (e.g. re-imported in a test) — without this, repeated calls would
    duplicate every log line once per extra handler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


def log_request(
    logger: logging.Logger,
    *,
    query: str,
    latency_ms: float,
    retrieval_count: int,
    sources: list[str],
) -> None:
    """
    Takes: the logger to write to, the user's query, end-to-end latency in
    milliseconds, how many chunks retrieval surfaced, and which source
    files they came from.
    Returns: nothing — writes one INFO-level log line.
    Use this: once per completed request (see app.graph.build_graph.run_turn),
    so every request's latency/retrieval_count/sources is recorded in the
    same shape and can be grepped for consistently.
    """
    logger.info(
        "request completed | query=%r latency_ms=%.1f retrieval_count=%d sources=%s",
        query,
        latency_ms,
        retrieval_count,
        sources,
    )


if __name__ == "__main__":
    # Tiny self-test: log one request with sample data, visually confirm
    # the line includes all four required fields in a consistent shape.
    demo_logger = get_logger(__name__)
    log_request(
        demo_logger,
        query="What is reciprocal rank fusion?",
        latency_ms=842.3,
        retrieval_count=3,
        sources=["rrf_intro.txt", "rrf_why_rank.txt"],
    )
