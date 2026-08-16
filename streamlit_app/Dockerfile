# Separate, much smaller image than the API's — this service only needs
# streamlit + requests, never app/'s ML dependencies (see app.py's docstring
# on why it never imports from app/).
FROM python:3.12-slim

WORKDIR /streamlit_app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
