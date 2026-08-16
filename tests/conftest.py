"""
Shared pytest setup for the smoke test suite.

Points CHROMA_PERSIST_DIR at a throwaway temp directory before any test
module imports app.config.settings (which reads env vars once, at import
time, into a module-level singleton) — so the suite never reads from or
writes into a developer's real ./data/chroma_db, and repeated local runs
don't accumulate test documents in the real index.
"""

import os
import tempfile

os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="unirag_test_chroma_"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "unirag_test_chunks")
