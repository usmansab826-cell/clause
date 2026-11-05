# Clause - API

FastAPI backend for **Clause**, a RAG assistant that answers questions about the shop handbook (`app/data/shop_handbook.md`).

> **Status:** early stage. The API server and basic endpoints work; the RAG pipeline is still being built.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- An API key for your chat model (Google Gemini by default, from [Google AI Studio](https://aistudio.google.com/apikey))
- Optional: [Ollama](https://ollama.com/) for local embeddings

## Setup

```bash
cd api
cp .env.example .env   # then fill in your values
uv sync
```

## Configuration

All settings are read from `.env`. `.env` is git-ignored, so your keys stay out of the repo.

### Chat model

| Variable           | Default                 | Description                                   |
|--------------------|-------------------------|-----------------------------------------------|
| `CHAT_PROVIDER`    | `google_genai`          | LangChain provider name                       |
| `CHAT_MODEL`       | `gemini-3.5-flash-lite` | Model ID                                      |
| `CHAT_TEMPERATURE` | `0.1`                   | Lower = more factual answers                  |
| `CHAT_BASE_URL`    | *(empty)*               | Custom endpoint, only if your provider needs one |
| `CHAT_API_KEY`     | *(empty)*               | **Required.** Your provider API key           |

To use another provider, install its LangChain package and update `.env`:

| Provider  | Install                       | `.env`                                                               |
|-----------|-------------------------------|----------------------------------------------------------------------|
| Anthropic | `uv add langchain-anthropic`  | `CHAT_PROVIDER=anthropic` `CHAT_MODEL=<model id>` `CHAT_API_KEY=sk-ant-...` |
| OpenAI    | `uv add langchain-openai`     | `CHAT_PROVIDER=openai` `CHAT_MODEL=<model id>` `CHAT_API_KEY=sk-...`        |

### Embeddings

| Variable               | Default   | Description                               |
|------------------------|-----------|-------------------------------------------|
| `EMBEDDING_PROVIDER`   | *(empty)* | e.g. `ollama` for local embeddings        |
| `EMBEDDING_MODEL`      | *(empty)* | e.g. `nomic-embed-text`                   |
| `EMBEDDING_BASE_URL`   | *(empty)* | e.g. `http://localhost:11434` for Ollama  |
| `EMBEDDING_API_KEY`    | *(empty)* | Only for hosted embedding providers       |
| `EMBEDDING_BATCH_SIZE` | `64`      | Chunks embedded per request               |

For local embeddings with Ollama, pull the model once:

```bash
ollama pull nomic-embed-text
```

Changing the embedding model is safe: Clause notices the change and re-embeds every chunk at the next start.

## Run

```bash
uv run fastapi dev
```

- API: http://localhost:8000
- Interactive docs (Swagger): http://localhost:8000/docs

The app entrypoint (`app.main:app`) is set in `pyproject.toml` under `[tool.fastapi]`.

## Endpoints

| Method | Path      | Description                      | Example response                                         |
|--------|-----------|----------------------------------|----------------------------------------------------------|
| GET    | `/`       | Welcome message                  | `{"message": "Welcome to Clause API", "docs": "/docs"}`  |
| GET    | `/health` | Health check                     | `{"status": "ok"}`                                       |
| GET    | `/hello`  | Greeting, optional `name` query  | `{"message": "Hello, World!"}`                           |

Try it:

```bash
curl http://localhost:8000/hello
curl "http://localhost:8000/hello?name=Rizwan"
```

## Project structure

```
api/
├── app/
│   ├── main.py              # FastAPI app and routes
│   ├── config/              # settings loaded from .env (planned)
│   ├── core/                # RAG logic (planned)
│   └── data/
│       └── shop_handbook.md # knowledge source for RAG
├── .env.example             # template for .env
└── pyproject.toml
```
