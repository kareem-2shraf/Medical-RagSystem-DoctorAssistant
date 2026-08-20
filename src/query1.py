import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from qdrant_client import QdrantClient
from qdrant_client.http import models 

def main():
    print("=== Launching Medical RAG Assistant ===")

    # 1. Load Embeddings (BGE Model)
    print("[1/5] Loading BGE-Small embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Connect to Qdrant Database
    print("[2/5] Connecting to local vector store...")
    client = QdrantClient(path="./qdrant_db")
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name="gdm_child_chunks", 
        embedding=embeddings
    )
    
    # 3. Set up Re-ranking Retriever with Metadata Filter & Threshold
    print("[3/5] Configuring Filters and FlashRank Re-ranker...")
    
    # فلترة النتائج بناءً على الـ Metadata اللي ضفناها في Ingest
    filter_condition = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.disease",
                match=models.MatchValue(value="Gestational Diabetes")
            )
        ]
    )

    # رفض الأسئلة خارج التخصص عن طريق Score Threshold
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.40, # تم رفع النسبة قليلاً لأن BGE أدق
            "k": 10,
            "filter": filter_condition
        }
    )
    
    compressor = FlashrankRerank(top_n=3)
    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # 4. Initialize Local LLM
    print("[4/5] Connecting to local Ollama (Qwen2.5)...")
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.1)

    # 5. Define System Prompt & Build Chain
    system_prompt = (
        "You are a strict, highly specialized clinical AI assistant. "
        "Your ONLY task is to answer the doctor's query based EXCLUSIVELY on the provided context. "
        "If the context is empty or does not contain the exact answer, you MUST reply verbatim: "
        "'I do not have enough information in the provided clinical guidelines to answer this question.' "
        "Under no circumstances should you guess, assume, or use outside knowledge.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(reranker_retriever, question_answer_chain)

    # 6. Execute Query
    # جرب تغير السؤال ده لسؤال بره الطب عشان تختبر الـ Threshold والـ Prompt
    question = "What is the fasting blood sugar target for gestational diabetes?"
    
    print(f"\n[5/5] Processing Query: '{question}'\n")

    response = rag_chain.invoke({"input": question})

    print("================ AI ASSISTANT ANSWER ================")
    print(response["answer"])
    print("=====================================================")

if __name__ == "__main__":
    main()
