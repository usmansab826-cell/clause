"""RAG-01: basic RAG over a Markdown file with an in-memory vector store.

Usage: python rag_01.py
"""

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = Path("shop_handbook.md")
CHAT_PROVIDER, CHAT_MODEL = "ollama", "gemma3:1b"
EMBED_PROVIDER, EMBED_MODEL = "ollama", "nomic-embed-text:latest"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 600, 100, 3

SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)


def build_index() -> InMemoryVectorStore:
    """Load -> chunk -> embed -> store."""
    doc = Document(page_content=DATA_PATH.read_text(encoding="utf-8"), metadata={"source": DATA_PATH.name})
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents([doc])
    embeddings = init_embeddings(model=EMBED_MODEL, provider=EMBED_PROVIDER)
    return InMemoryVectorStore.from_documents(chunks, embedding=embeddings)


def answer(question: str, store: InMemoryVectorStore, llm) -> str:
    """Retrieve -> augment -> generate."""
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm.invoke(messages).text


def main() -> None:
    store = build_index()
    llm = init_chat_model(model=CHAT_MODEL, model_provider=CHAT_PROVIDER, temperature=0.1)
    print("Ask about the shop handbook. Type 'exit' to quit.")
    try:
        while (question := input("You: ").strip()).lower() not in {"exit", "quit"}:
            if question:
                print(f"AI: {answer(question, store, llm)}")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
