"""RAG-03: RAG over a PDF with a persistent Qdrant vector store.

Usage:
    python rag_03.py            # chat (indexes the PDF on the first run)
    python rag_03.py --reindex  # rebuild the collection, e.g. after the PDF changes
"""

import sys
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DATA_PATH = Path("shop_handbook.pdf")
CHAT_PROVIDER, CHAT_MODEL = "ollama", "gemma3:1b"
EMBED_PROVIDER, EMBED_MODEL = "ollama", "nomic-embed-text:latest"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 600, 100, 3
SCORE_THRESHOLD = 0.5  # cosine similarity; tune for your embedding model

COLLECTION = "shop_handbook"
QDRANT_URL = None  # e.g. "http://localhost:6333" for the Docker server
QDRANT_PATH = "./qdrant_data"  # local mode, used when QDRANT_URL is None

NO_ANSWER = "I don't know. The handbook doesn't cover that."
SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)


def load_pdf(path: Path) -> list[Document]:
    return [
        Document(page_content=text, metadata={"source": path.name, "page": i})
        for i, page in enumerate(PdfReader(path).pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]


def index(client: QdrantClient, embeddings: Embeddings) -> int:
    """(Re)build the collection from the PDF and return the number of chunks stored."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(load_pdf(DATA_PATH))

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=len(embeddings.embed_query("probe")), distance=Distance.COSINE),
    )
    QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings).add_documents(chunks)
    return len(chunks)


def answer(question: str, store: QdrantVectorStore, llm) -> tuple[str, list[int]]:
    # Qdrant drops chunks below the threshold, so off-topic questions never reach the LLM.
    results = store.similarity_search_with_score(question, k=TOP_K, score_threshold=SCORE_THRESHOLD)
    if not results:
        return NO_ANSWER, []

    context = "\n\n".join(doc.page_content for doc, _ in results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm.invoke(messages).text, sorted({doc.metadata["page"] for doc, _ in results})


def main() -> None:
    client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)
    embeddings = init_embeddings(model=EMBED_MODEL, provider=EMBED_PROVIDER)
    try:
        if "--reindex" in sys.argv or not client.collection_exists(COLLECTION):
            print(f"Indexed {index(client, embeddings)} chunks from {DATA_PATH.name}")

        store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
        llm = init_chat_model(model=CHAT_MODEL, model_provider=CHAT_PROVIDER, temperature=0.1)

        print("Ask about the shop handbook. Type 'exit' to quit.")
        while (question := input("You: ").strip()).lower() not in {"exit", "quit"}:
            if not question:
                continue
            text, pages = answer(question, store, llm)
            print(f"AI: {text}")
            if pages:
                print(f"Sources: pages {pages}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
