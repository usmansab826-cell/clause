from langchain_core.documents import Document
from Langchain_community.documents_loders import PyPDFLoader
from Langchain_text_spliters import RecursiveCharacterTextSplitter
from Langchain_qdrant import Qdrant
from pypdf import PdfReader
from qdrant_client.models import Distance, VectorParams
from core.config import get_settings,get_embeddings

def load(path: str | None = None):

setting = get_settings()

file = path(path or settings.data_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    length_function=["\n\n","\n", " ", ""]

)
chunks = text_splitter.split_documents(documents=load(path=file))

embed = get_embeddings(settings)

client = get_qdrant_client(settings)

try:
    if client.get_collection(collection_name=settings.collection_name):
        client.delete_collection(collection_name=settings.collection_name)

   client.create_collection(collection_name=settings.collection_name, vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE))

   vector_store = Qdrantvectorstore(client=client, collection_name=settings.collection_name, embeddings=embed)

except:
    client.close()

    return len(chunks)

if _name_ == "_main_":
    ingest()
