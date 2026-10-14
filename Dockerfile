FROM python:3.12-slim

# Reflex downloads bun (frontend package manager) at the first build; it needs curl and unzip.
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch for the assistant's embeddings (the same library and version the pipeline
# builds the RAG index with). The CPU build: the default wheel pulls about 2 GB of CUDA libraries never used here.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The embedding model goes into the image, so the hosted app never downloads anything at run time. Its name is read
# from the configuration (app.rag.embedding_model in conf/app.yaml), the one place it is set.
ENV HF_HOME=/opt/hf
COPY smogcast/app/conf/app.yaml /tmp/app.yaml
RUN python -c "import yaml; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(yaml.safe_load(open('/tmp/app.yaml'))['app']['rag']['embedding_model'])" \
    && rm /tmp/app.yaml
ENV HF_HUB_OFFLINE=1

COPY . .

# The frontend is built into the image (bun, the JavaScript packages and the compiled pages in /app/.web), so the
# hosted app downloads nothing when it starts. Measured on 1 CPU / 2 GiB: this step takes about 30 s and the app then
# answers 11 s after start instead of 20 s.
RUN reflex export --frontend-only --no-zip

# What the hosting runs: frontend and backend (websocket) on ONE port, 3000. `reflex run --env prod` is the documented
# way; it builds the pages once more at start from the packages already in the image.
# Local development overrides this command in docker-compose.yml (dev server: frontend 3000, backend 8000, hot reload).
EXPOSE 3000
# Ties the package in GitHub Container Registry to this repository: the package shows on the repo's page and the
# repo's workflows (deploy.yml) may push new images to it.
LABEL org.opencontainers.image.source="https://github.com/adamalama151212-code/smog-cast-app"
CMD ["reflex", "run", "--env", "prod", "--single-port", "--backend-host", "0.0.0.0"]
