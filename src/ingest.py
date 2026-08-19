import os
import requests
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db():
    # 1. Download PDF
    url = "https://gynerisq.fr/wp-content/uploads/2018/12/ACOG-2018_Diabete-gestationnel.pdf"
    pdf_path = "../data/ACOG_GDM_190.pdf"
    
    os.makedirs("../data", exist_ok=True)
    if not os.path.exists(pdf_path):
        print("Downloading PDF...")
        response = requests.get(url)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)

    # 2. Parse PDF to Markdown
    print("Parsing PDF to Markdown...")
    md_text = pymupdf4llm.to_markdown(pdf_path)

    # 3. Chunking
    print("Chunking Text...")
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(md_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(md_header_splits)

    for chunk in splits:
        chunk.metadata['source'] = "ACOG Practice Bulletin No. 190"

    # 4. Embeddings & Indexing
    print("Loading Embedding Model and Building DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_dir = "../chroma_medical_db"
    
    vector_db = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_model, 
        persist_directory=db_dir
    )
    print(f"Ingestion Complete! {len(splits)} chunks saved to DB.")

if __name__ == "__main__":
    build_vector_db()