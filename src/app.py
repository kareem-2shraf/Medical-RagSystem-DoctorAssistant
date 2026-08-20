import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from qdrant_client import QdrantClient

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺", layout="centered")

st.title("🩺 Medical RAG Assistant")
st.caption("Gestational Diabetes Clinical Support (Powered by Local AI & BGE-Small)")

# --- 2. CACHE THE RAG PIPELINE ---
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    client = QdrantClient(path="./qdrant_db")
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name="gdm_child_chunks", 
        embedding=embeddings
    )
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    compressor = FlashrankRerank(top_n=3)
    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.1)
    
    system_prompt = (
        "You are a specialized Gestational Diabetes Assistant for doctors. "
        "Use the following pieces of retrieved medical context to answer the question professionally. "
        "If the answer is not in the context, state clearly: 'I cannot find this in the clinical guidelines.' "
        "Do not guess or hallucinate facts."
        "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(reranker_retriever, question_answer_chain)

with st.spinner("Initializing Vector Database & FlashRank Reranker..."):
    rag_chain = load_rag_pipeline()

# --- 3. MANAGE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages (including stored source context)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Retrieved Source Context"):
                for idx, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.info(doc.page_content)

# --- 4. HANDLE USER INPUT ---
if prompt := st.chat_input("Ask a clinical question about Gestational Diabetes..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and re-ranking medical guidelines..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            sources = response.get("context", [])
            
            st.markdown(answer)
            
            # Display source chunks for verification
            if sources:
                with st.expander("📚 View Retrieved Source Context"):
                    for idx, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.info(doc.page_content)
            
    # Store answer and sources in session memory
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })