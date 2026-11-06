"""RAG-02: RAG over a PDF with similarity scores; only relevant chunks reach the LLM.

Usage: python rag_02.py
"""

from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_PATH = Path("shop_handbook.pdf")
CHAT_PROVIDER, CHAT_MODEL = "ollama", "gemma3:1b"
EMBED_PROVIDER, EMBED_MODEL = "ollama", "nomic-embed-text:latest"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 600, 100, 3
SCORE_THRESHOLD = 0.5  # cosine similarity (higher = more relevant); tune for your embedding model

NO_ANSWER = "I don't know. The handbook doesn't cover that."
SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)


def load_pdf(path: Path) -> list[Document]:
    """One Document per non-empty page, with 1-based page numbers."""
    return [
        Document(page_content=text, metadata={"source": path.name, "page": i})
        for i, page in enumerate(PdfReader(path).pages, start=1)
        if (text := page.extract_text() or "").strip()
    ]


def build_index() -> InMemoryVectorStore:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(load_pdf(DATA_PATH))
    embeddings = init_embeddings(model=EMBED_MODEL, provider=EMBED_PROVIDER)
    return InMemoryVectorStore.from_documents(chunks, embedding=embeddings)


def answer(question: str, store: InMemoryVectorStore, llm) -> tuple[str, list[int]]:
    """Return the answer and the source pages it was based on."""
    results = store.similarity_search_with_score(question, k=TOP_K)
    relevant = [doc for doc, score in results if score >= SCORE_THRESHOLD]
    if not relevant:
        return NO_ANSWER, []

    context = "\n\n".join(d.page_content for d in relevant)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm.invoke(messages).text, sorted({d.metadata["page"] for d in relevant})


def main() -> None:
    store = build_index()
    llm = init_chat_model(model=CHAT_MODEL, model_provider=CHAT_PROVIDER, temperature=0.1)
    print("Ask about the shop handbook. Type 'exit' to quit.")
    try:
        while (question := input("You: ").strip()).lower() not in {"exit", "quit"}:
            if not question:
                continue
            text, pages = answer(question, store, llm)
            print(f"AI: {text}")
            if pages:
                print(f"Sources: pages {pages}")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
