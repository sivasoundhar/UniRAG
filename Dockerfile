# Slim Python base — matches the project's dev venv (3.12) and keeps the
# image small; no compiler toolchain needed at runtime beyond pip installs.
FROM python:3.12-slim

# tesseract-ocr: app/loaders/ocr_loader.py shells out to the tesseract
# binary via pytesseract, which isn't a pure-Python package pip can install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies layer first, separate from app code, so `docker build` only
# re-runs pip install when requirements.txt actually changes, not on every
# code edit.
COPY requirements.txt .
# --trusted-host: some networks' TLS-inspecting proxies (e.g. antivirus
# HTTPS scanning) break certificate verification for pypi.org from inside
# Docker's network path specifically, even when host-side pip installs are
# unaffected — the same class of issue can hit huggingface.co/api.groq.com
# directly on the host too.
# Scoped to these two package-index hosts, build time only — does not
# affect the app's own runtime HTTPS calls (Groq/Ollama) at all.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

COPY app/ ./app/

# data/uploads and data/chroma_db are volume-mounted at runtime (see
# docker-compose.yml) — created here too so a fresh container run without
# the compose volumes still has somewhere to write on first request.
RUN mkdir -p data/uploads data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
