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
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🩺 MEDRAG Clinical Support")
st.caption("A medical AI assistant grounded in Gestational Diabetes guidelines.")
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
    
    # 3. Stricter Similarity Threshold (Filter out loose keyword matches)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.55,  # Raised from 0.40 to prevent semantic drift
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
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.0)
    
    # 6. Hardened System Prompt with Few-Shot Refusal Demonstration
    system_prompt = (
        "You are an evidence-grounded medical AI assistant specialized ONLY in Gestational Diabetes guidelines.\n"
        "RULES:\n"
        "1. Answer ONLY using facts directly mentioned in the provided Context.\n"
        "2. If the Context does not explicitly mention the exact topic or treatment requested in the Question, "
        "you MUST reply with ONLY this exact sentence: 'I do not have enough evidence in the provided clinical guidelines to answer this question safely.'\n"
        "3. NEVER use outside medical knowledge.\n\n"
        "--- EXAMPLE 1 ---\n"
        "Context: ACOG guidelines state fasting blood sugar should be below 95 mg/dL.\n"
        "Question: What is the fasting glucose target?\n"
        "Answer: The target fasting blood glucose level is below 95 mg/dL [Document: ACOG, Section: Targets].\n\n"
        "--- EXAMPLE 2 ---\n"
        "Context: Guidelines discuss insulin dosing for pregnant women with diabetes.\n"
        "Question: What chemotherapy is recommended for breast cancer in pregnancy?\n"
        "Answer: I do not have enough evidence in the provided clinical guidelines to answer this question safely.\n\n"
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

# Helper function to render Evidence Panel cleanly
def render_evidence_panel(docs, expanded=False):
    with st.expander("📊 Evidence Panel: Retrieved Sources & Metadata", expanded=expanded):
        for idx, doc in enumerate(docs, 1):
            st.markdown(f"### 📑 Chunk #{idx}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**📄 Document:** `{doc.metadata.get('source', 'Unknown')}`")
                st.markdown(f"**🏢 Organization:** `{doc.metadata.get('organization', 'N/A')}`")
            with col2:
                st.markdown(f"**📌 Section:** `{doc.metadata.get('Section', 'General')}`")
                st.markdown(f"**🎯 Category:** `{doc.metadata.get('category', 'N/A')}`")
            with col3:
                st.markdown(f"**📑 Subsection:** `{doc.metadata.get('Subsection', 'N/A')}`")
                st.markdown(f"**🩺 Disease:** `{doc.metadata.get('disease', 'N/A')}`")
            
            st.markdown("**Retrieved Guideline Excerpt:**")
            st.info(doc.page_content)
            
            with st.expander(f"🔍 View Raw Metadata Schema (Chunk #{idx})"):
                st.json(doc.metadata)
            st.divider()

# --- 3. MANAGE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_evidence_panel(msg["sources"], expanded=False)

# --- 4. HANDLE USER INPUT ---
if user_prompt := st.chat_input("Enter a clinical query (e.g., 'What is the fasting blood sugar target?')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing clinical guidelines..."):
            response = rag_chain.invoke({"input": user_prompt})
            answer = response.get("answer", "")
            retrieved_sources = response.get("context", [])
            
            st.markdown(answer)
            
            if retrieved_sources:
                render_evidence_panel(retrieved_sources, expanded=True)
            else:
                st.warning("⚠️ Retrieval Confidence Threshold not met. No relevant chunks found.")
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": retrieved_sources
    })