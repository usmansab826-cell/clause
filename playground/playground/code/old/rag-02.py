from pathlib import Path

from langchain.chat_models import init_chat_model # LLM
from langchain.embeddings import init_embeddings  # Embeddings

from langchain_core.documents import Document # .txt, .md
# from langchain_community.document_loaders import PyPDFLoader # .pdf

from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore # vector store


DATA_PATH = "shop_handbook.md"
CHAT_PROVIDER = "ollama"
CHAT_MODEL = "gemma3:1b"
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "nomic-embed-text:latest"
TOP_K = 3

SYSTEM_PROMPT = (
    "You are a helpful Clause Shop assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you don't know."
)

# Indexing
# 1. Load
text = Path(DATA_PATH).read_text(encoding="utf-8")
doc = Document(page_content=text, metadata={"source": "shop_handbook.md"})

# 2. Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents([doc])

# 3. Embed
embedd = init_embeddings(
    model=EMBEDDING_MODEL,
    provider=EMBEDDING_PROVIDER
)

# 4. Store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embedd
)

# Retrieval-Augmented Generation (RAG)
# LLM
llm = init_chat_model(
    model=CHAT_MODEL,
    model_provider=CHAT_PROVIDER,
    temperature=0.1  # factual answers, less guessing
)

query = input("You: ").strip()

# Retrieve
# retrieved_docs = vector_store.similarity_search(query, k=TOP_K) # Documents

results = vector_store.similarity_search_with_score(query, k=TOP_K) # [(Document, float), ...]

# print("\n --- Retrieved Chunks ---")
# for rank, (doc, score) in enumerate(results, start=1):
#     print(f"Rank {rank}: Score {score:.4f}")
#     print(f"Content: {doc.page_content}\n")
#     print(f"Metadata: {doc.metadata}\n")
# print("-------------------------\n")

retrieved_docs = [doc for doc, _ in results] # Documents
context = "\n\n".join(d.page_content for d in retrieved_docs) # string object

# Generate
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
]
print("AI: ", llm.invoke(messages).text)
