import uuid
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pymupdf4llm
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever

# ---------------------------------------------------------
# DYNAMIC PATH RESOLUTION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PDF_PATH = PROJECT_ROOT / "data" / "ACOG-2018_Diabete-gestationnel.pdf"
DB_PATH = PROJECT_ROOT / "qdrant_db"
DOCSTORE_PATH = PROJECT_ROOT / "parent_docstore"

# ---------------------------------------------------------
# STEP 1: Parse PDF into Markdown
# ---------------------------------------------------------
print(f"Parsing PDF from: {PDF_PATH}")
gdm_markdown = pymupdf4llm.to_markdown(str(PDF_PATH))

headers_to_split_on = [
    ("#", "Document Title"),
    ("##", "Section"),
    ("###", "Subsection"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
)
parent_docs = markdown_splitter.split_text(gdm_markdown)

# ---------------------------------------------------------
# STEP 2: Configure Embedding & Splitter
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=40
)

# ---------------------------------------------------------
# STEP 3: Setup Qdrant & Auto-Create Collection if Missing
# ---------------------------------------------------------
client = QdrantClient(path=str(DB_PATH))
collection_name = "gdm_child_chunks"

# Check and create collection if it does not exist yet
if not client.collection_exists(collection_name=collection_name):
    print(f"Creating Qdrant collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 dims for MiniLM
    )

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)

# LocalFileStore persists parent docs on disk across python executions
fs = LocalFileStore(str(DOCSTORE_PATH))
store = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=None 
)

print("Ingesting documents into vector database and local docstore...")
retriever.add_documents(parent_docs, ids=None)

print(f"\nIngestion successful! Data stored at:\n - Vector DB: {DB_PATH}\n - Docstore: {DOCSTORE_PATH}")