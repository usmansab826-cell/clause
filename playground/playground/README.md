# Clause - Playground

A scratch space for trying out RAG (Retrieval-Augmented Generation) ideas for **Clause**, the Clause Shop support assistant, before they move into the real backend in [`../api`](../api/README.md).

Everything here runs locally with [Ollama](https://ollama.com/): no API key needed.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally

Pull the two models the scripts use:

```bash
ollama pull gemma3:4b          # chat model
ollama pull nomic-embed-text   # embedding model
```

## Setup

```bash
cd playground
uv sync --active
```

This installs LangChain, the Ollama integration, the text splitters, `python-dotenv`, and `ipykernel` (so you can also use the environment as a Jupyter kernel).

## Scripts

| File                     | What it does                                                               |
|--------------------------|----------------------------------------------------------------------------|
| `code/rag-01.py`         | A minimal, single-question RAG pipeline over the shop handbook             |
| `code/rag-02.py`         | Next step (empty for now)                                                  |
| `code/shop_handbook.md`  | The knowledge source: Clause Shop's customer policy handbook               |
| `app/main.py`            | Placeholder entry point; prints a greeting                                 |

### `rag-01.py`: basic RAG in one file

The script walks through the whole RAG flow, top to bottom:

1. **Load** `shop_handbook.md` as a LangChain `Document`
2. **Chunk** it with `RecursiveCharacterTextSplitter` (600 characters, 100 overlap)
3. **Embed** the chunks with `nomic-embed-text` via Ollama
4. **Store** them in an `InMemoryVectorStore`
5. **Retrieve** the top 3 chunks most similar to your question
6. **Generate** an answer with `gemma3:1b`, told to answer only from that context

Run it from inside `code/`, because it reads `shop_handbook.md` by relative path:

```bash
cd code
uv run --active rag-01.py
```

Then type a question:

```text
You: How long do I have to return a phone?
AI:  Electronics have a 7-day refund window...
```

The script answers one question and exits. Its settings (models, `TOP_K`, system prompt) are constants at the top of the file, so it's easy to change them and rerun.

## Configuration

`rag-01.py` uses hard-coded settings and does not read `.env`. The `.env.example` file is a template for later scripts, which will load settings from `.env` the same way the API does:

```bash
cp .env.example .env
```

| Variable               | Example                    | Description                              |
|------------------------|----------------------------|------------------------------------------|
| `CHAT_PROVIDER`        | `ollama`                   | LangChain provider name (`fake` for offline tests) |
| `CHAT_MODEL`           | `gemma3:4b`                | Chat model ID                            |
| `CHAT_TEMPERATURE`     | `0.1`                      | Lower = more factual answers             |
| `CHAT_BASE_URL`        | *(empty)*                  | Custom endpoint, only if needed          |
| `CHAT_API_KEY`         | *(empty)*                  | Only for hosted providers                |
| `EMBEDDING_PROVIDER`   | `ollama`                   | Embedding provider (`fake` for offline tests) |
| `EMBEDDING_MODEL`      | `nomic-embed-text`         | Embedding model ID                       |
| `EMBEDDING_BASE_URL`   | `http://localhost:11434`   | Ollama's default address                 |
| `EMBEDDING_API_KEY`    | *(empty)*                  | Only for hosted providers                |
| `EMBEDDING_BATCH_SIZE` | `64`                       | Chunks embedded per request              |

`.env` is git-ignored, so your keys stay out of the repo.

## Project structure

```
playground/
├── app/
│   ├── main.py              # placeholder entry point
│   ├── config/              # settings (planned)
│   ├── core/                # RAG logic (planned)
│   └── data/
│       └── shop_handbook.md # copy of the handbook
├── code/
│   ├── rag-01.py            # basic RAG, step by step
│   ├── rag-02.py            # next experiment (empty)
│   └── shop_handbook.md     # handbook used by the scripts
├── .env.example             # template for .env
└── pyproject.toml
```
