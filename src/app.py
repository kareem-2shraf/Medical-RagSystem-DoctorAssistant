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

# --- 1. PAGE SETUP & UI BRANDING ---
st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺", layout="wide")

# Persistent Sidebar for Safety & Guardrails
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80) # Placeholder medical icon
    st.title("System Guardrails")
    st.warning(
        "**CLINICAL SAFETY DISCLAIMER**\n\n"
        "This system supports — never replaces — clinical judgment. "
        "Outputs are guideline-grounded, not diagnostic."
    )
    st.info(
        "**Core Philosophy:**\n"
        "Every clinical recommendation must trace back to an official, citable source. "
        "No private or credential-gated data is used."
    )
    st.divider()
    st.caption("Powered by Local AI (Qwen2.5) & BGE-Small")

st.title("🩺 Gestational Diabetes Clinical Support")
st.caption("Evidence-Grounded Retrieval-Augmented Generation (RAG)")
st.divider()

# --- 2. CACHE THE RAG PIPELINE ---
@st.cache_resource
def load_rag_pipeline():
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 2. Vector Store
    client = QdrantClient(path="./qdrant_db")
    vectorstore = QdrantVectorStore(
        client=client, 
        collection_name="gdm_child_chunks", 
        embedding=embeddings
    )
    
    # 3. Retriever with Confidence Thresholds
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.40, # Blocks out-of-scope queries
            "k": 10
        }
    )
    
    # 4. Reranker
    compressor = FlashrankRerank(top_n=3)
    reranker_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # 5. LLM
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.0) # Temp 0.0 for strict factual output
    
    # 6. Strict Prompt with Citation Mechanics
    system_prompt = (
        "You are a strict, highly specialized clinical AI assistant. "
        "Your ONLY task is to answer the doctor's query based EXCLUSIVELY on the provided context. "
        "If the context is empty or does not contain the exact answer, you MUST reply verbatim: "
        "'I do not have enough evidence in the provided clinical guidelines to answer this question safely.'\n\n"
        "CITATION MECHANICS:\n"
        "You must append a citation to every clinical claim you make. "
        "Format citations strictly as: [Document: <name>, Section: <title>, Page: <number>].\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(reranker_retriever, question_answer_chain)

with st.spinner("Initializing Vector Database & Guardrails..."):
    rag_chain = load_rag_pipeline()

# --- 3. MANAGE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you with gestational diabetes guidelines today?"}]

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render the Evidence Panel for previous assistant messages
        if "sources" in msg and msg["sources"]:
            with st.expander("📊 Evidence Panel: Retrieved Source Context"):
                for idx, doc in enumerate(msg["sources"], 1):
                    meta = doc.metadata
                    source_name = meta.get("source", "Unknown Document")
                    disease = meta.get("disease", "N/A")
                    st.markdown(f"**Source {idx}:** `{source_name}` (Category: *{disease}*)")
                    st.info(doc.page_content)

# --- 4. HANDLE USER INPUT ---
if prompt := st.chat_input("Enter a clinical query (e.g., 'What is the fasting blood sugar target?')..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical guidelines..."):
            
            # Execute Pipeline
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            sources = response.get("context", [])
            
            # Display Answer
            st.markdown(answer)
            
            # Display Evidence Panel UI for live verification
            if sources:
                with st.expander("📊 Evidence Panel: Retrieved Source Context", expanded=True):
                    for idx, doc in enumerate(sources, 1):
                        meta = doc.metadata
                        # Extracting metadata safely
                        source_name = meta.get("source", "Unknown Document")
                        disease = meta.get("disease", "N/A")
                        
                        st.markdown(f"**Source {idx}:** `{source_name}` (Category: *{disease}*)")
                        st.info(doc.page_content)
            else:
                st.warning("⚠️ Retrieval Confidence Threshold not met. No relevant chunks found.")
            
    # Store in memory
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
