import os
from pathlib import Path
import pandas as pd
from typing import List, Dict
import pymupdf4llm

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# ---------------------------------------------------------
# DYNAMIC PATH RESOLUTION (Fixes file not found errors)
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PDF_PATH = PROJECT_ROOT / "data" / "ACOG-2018_Diabete-gestationnel.pdf"

# ---------------------------------------------------------
# 1. BENCHMARK DATASET SETUP (ACOG 2018 Guidelines)
# ---------------------------------------------------------
EVAL_DATASET = [
    {
        "id": "Q1",
        "query": "What are the target blood glucose levels for fasting and postprandial monitoring in GDM?",
        "must_contain": ["95", "140", "120", "fasting"],
    },
    {
        "id": "Q2",
        "query": "What is the preferred first-line pharmacologic treatment for gestational diabetes?",
        "must_contain": ["insulin", "preferred", "first-line"],
    },
    {
        "id": "Q3",
        "query": "When should postpartum screening for diabetes be conducted and what test is recommended?",
        "must_contain": ["4–12 weeks", "75-g", "ogtt", "postpartum"],
    },
    {
        "id": "Q4",
        "query": "What estimated fetal weight threshold justifies counseling for a scheduled cesarean delivery?",
        "must_contain": ["4,500", "cesarean", "fetal weight"],
    },
    {
        "id": "Q5",
        "query": "What are the exercise recommendations for managing gestational diabetes?",
        "must_contain": ["30 minutes", "150 minutes", "walking"],
    }
]

# ---------------------------------------------------------
# 2. RETRIEVER PIPELINE BUILDER
# ---------------------------------------------------------
def build_parent_retriever(
    pdf_path: str, 
    model_name: str, 
    collection_name: str, 
    chunk_size: int = 200, 
    chunk_overlap: int = 40
) -> ParentDocumentRetriever:
    """Builds and populates a ParentDocumentRetriever with specified embedding & chunk config."""
    
    print(f"  -> Parsing PDF for {collection_name}...")
    gdm_markdown = pymupdf4llm.to_markdown(pdf_path)
    
    headers_to_split_on = [("#", "Document Title"), ("##", "Section"), ("###", "Subsection")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    parent_docs = markdown_splitter.split_text(gdm_markdown)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Temporary In-Memory Qdrant Collection for rapid evaluation
    client = QdrantClient(location=":memory:")
    
    # FIX: Ensure the collection is created before QdrantVectorStore connects to it!
    # Both MiniLM and BGE-Small use 384-dimensional embeddings
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    # Create an empty temporary document store in memory for this evaluation run
    docstore_dict = {}
    from langchain_core.stores import InMemoryStore
    docstore = InMemoryStore()

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=None
    )
    
    print("  -> Ingesting chunks...")
    retriever.add_documents(parent_docs, ids=None)
    return retriever

# ---------------------------------------------------------
# 3. EVALUATION METRICS IMPLEMENTATION
# ---------------------------------------------------------
def evaluate_retrieval(retriever: ParentDocumentRetriever, top_k: int = 3) -> Dict[str, float]:
    """Computes Precision@k, Recall@k, and Hit Rate@k across the benchmark test set."""
    precision_scores = []
    recall_scores = []
    hits = 0

    for item in EVAL_DATASET:
        retrieved_docs = retriever.invoke(item["query"])[:top_k]
        
        relevant_count = 0
        for doc in retrieved_docs:
            content_lower = doc.page_content.lower()
            matches = [kw.lower() in content_lower for kw in item["must_contain"]]
            if sum(matches) >= len(item["must_contain"]) // 2:
                relevant_count += 1

        precision_at_k = relevant_count / top_k
        recall_at_k = min(1.0, relevant_count / 1.0)

        precision_scores.append(precision_at_k)
        recall_scores.append(recall_at_k)
        if relevant_count > 0:
            hits += 1

    return {
        f"Precision@{top_k}": round(sum(precision_scores) / len(precision_scores), 4),
        f"Recall@{top_k}": round(sum(recall_scores) / len(recall_scores), 4),
        f"Hit_Rate@{top_k}": round(hits / len(EVAL_DATASET), 4)
    }

# ---------------------------------------------------------
# 4. COMPARATIVE EXPERIMENTAL RUNNER
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Evaluation using PDF: {PDF_PATH}\n")
    
    models_to_test = [
        ("Model A (MiniLM)", "sentence-transformers/all-MiniLM-L6-v2"),
        ("Model B (BGE-Small)", "BAAI/bge-small-en-v1.5")
    ]

    results = []

    for label, model_id in models_to_test:
        print(f"=== Evaluating {label} ===")
        retriever = build_parent_retriever(
            pdf_path=str(PDF_PATH),
            model_name=model_id,
            collection_name=f"test_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
            chunk_size=200,
            chunk_overlap=40
        )
        
        metrics_k3 = evaluate_retrieval(retriever, top_k=3)
        metrics_k5 = evaluate_retrieval(retriever, top_k=5)
        
        results.append({
            "Model": label,
            "Chunk Size/Overlap": "200/40",
            "P@3": metrics_k3["Precision@3"],
            "R@3": metrics_k3["Recall@3"],
            "P@5": metrics_k5["Precision@5"],
            "R@5": metrics_k5["Recall@5"],
            "Hit Rate@3": metrics_k3["Hit_Rate@3"]
        })

    # Output Summary Comparison Table
    df_results = pd.DataFrame(results)
    print("\n=== RETRIEVAL EVALUATION RESULTS ===")
    print(df_results.to_markdown(index=False))
    # You will import your specific LLM here (e.g., OpenAI or Ollama)

# 1. Define the LLM (This is the "Brain" that will read the text)
llm = ... # We will fill this in next!

# 2. Write the Medical System Prompt
system_prompt = (
    "You are a specialized Gestational Diabetes Assistant for doctors. "
    "Use the following pieces of retrieved medical context to answer the question. "
    "If the answer is not in the context, say 'I cannot find this in the guidelines.' "
    "Do not hallucinate or guess."
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 3. Connect the Retriever, the Prompt, and the LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Ask a question!
response = rag_chain.invoke({"input": "What is the first-line medication for GDM?"})
print(response["answer"])