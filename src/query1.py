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

def main():
    print("=== Launching Medical RAG Assistant ===")

    # 1. Load Embeddings
    print("[1/5] Loading BGE-Small embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # 2. Connect to Qdrant Database
    print("[2/5] Connecting to local vector store...")
    client = QdrantClient(path="./qdrant_db")
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name="gdm_child_chunks", 
        embedding=embeddings
    )
    
    # 3. Set up Re-ranking Retriever
    print("[3/5] Initializing FlashRank Re-ranker (Top 10 -> Top 3)...")
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
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
        "You are a specialized Gestational Diabetes Assistant for doctors. "
        "Use the following pieces of retrieved medical context to answer the question professionally. "
        "If the answer is not in the context, state clearly: 'I cannot find this in the clinical guidelines.' "
        "Do not guess or hallucinate facts."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(reranker_retriever, question_answer_chain)

    # 6. Execute Query
    question = "What is the fasting blood sugar target for gestational diabetes?"
    print(f"\n[5/5] Processing Query: '{question}'\n")

    response = rag_chain.invoke({"input": question})

    print("================ AI ASSISTANT ANSWER ================")
    print(response["answer"])
    print("=====================================================")

if __name__ == "__main__":
    main()