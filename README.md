# 🩺 MedRAG — Gestational Diabetes Clinical Assistant

A fully local **Retrieval-Augmented Generation (RAG)** assistant that answers clinical questions about gestational diabetes (GDM) strictly from the **ACOG 2018 guideline**. Built during a one-week AI hackathon.

Every answer is grounded in retrieved guideline text, shown to the doctor alongside the response, and the assistant refuses to answer when the guideline doesn't cover the question.

---

## Why this project

- Gestational diabetes affects roughly **1 in 6 live births** worldwide (IDF Diabetes Atlas) and raises the risk of preeclampsia, cesarean delivery, macrosomia, and neonatal hypoglycemia.
- Clinical guidelines are long PDFs that are slow to search at the point of care.
- Generic chatbots can hallucinate medical facts. MedRAG answers **only** from verified guideline text and runs **100% locally**, so no patient-adjacent queries leave the machine.

---

## Architecture

```
PDF guideline
   │  pymupdf4llm
   ▼
Markdown
   │  MarkdownHeaderTextSplitter  (#, ##, ###  →  parent documents)
   │  RecursiveCharacterTextSplitter (200 chars / 40 overlap  →  child chunks)
   ▼
Hybrid hierarchical chunks (ParentDocumentRetriever)
   │  BAAI/bge-small-en-v1.5 embeddings
   ▼
Qdrant vector store (local, on disk)
   │  top-10 similarity search
   ▼
FlashRank reranker  →  top-3 chunks
   │
   ▼
Qwen2.5:0.5B via Ollama (temperature 0.1)
   │
   ▼
Streamlit chat UI  (answer + expandable source chunks)
```

### Technology choices

| Component | Chosen | Alternatives considered | Why |
|---|---|---|---|
| Chunking | Hybrid: Markdown-header parents + recursive child chunks | Plain fixed-size chunking | Preserves clinical section hierarchy and context |
| Embeddings | `BAAI/bge-small-en-v1.5` | `all-MiniLM-L6-v2`, a medical-domain embedding model | Highest hit rate in evaluation (~73% retrieval accuracy) |
| Vector store | Qdrant (local) | Chroma, FAISS | Native LangChain support, persistent local index |
| Retrieval | Top-10 → FlashRank rerank → top-3 | Plain top-k similarity | Higher precision, less noise reaching the LLM |
| Generation | `qwen2.5:0.5b` via Ollama | Cloud LLM APIs | Privacy, zero cost, offline demo |
| UI | Streamlit | Custom front-end | Fast to build within hackathon time |

---

## Project structure

```
.
├── data/
│   └── ACOG-2018_Diabete-gestationnel.pdf   # source guideline
├── src/
│   ├── ingest1.py    # parse PDF → hybrid chunking → embed → Qdrant + docstore
│   ├── query1.py     # CLI: ask a single question through the RAG chain
│   ├── app.py        # Streamlit chat interface
│   └── Eval.py       # retrieval benchmark (Precision@k, Recall@k, Hit Rate@k)
├── qdrant_db/        # created by ingestion
├── parent_docstore/  # created by ingestion
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### 2. Install dependencies

```bash
pip install streamlit pandas tabulate pymupdf4llm qdrant-client \
    langchain langchain-classic langchain-community langchain-core \
    langchain-huggingface langchain-qdrant langchain-ollama \
    langchain-text-splitters sentence-transformers flashrank
```

### 3. Pull the LLM

```bash
ollama pull qwen2.5:0.5b
```

### 4. Add the guideline PDF

Place it at `data/ACOG-2018_Diabete-gestationnel.pdf`.

---

## Usage

> Run all commands from the **project root** so the relative `./qdrant_db` path resolves correctly.

**1. Ingest the guideline** (run once)

```bash
python src/ingest1.py
```

> ⚠️ The embedding model used at ingestion **must match** the one used at query time. Both `ingest1.py` and the query/app scripts should use `BAAI/bge-small-en-v1.5`.

**2. Ask a question from the terminal**

```bash
python src/query1.py
```

**3. Launch the chat app**

```bash
streamlit run src/app.py
```

**4. Run the retrieval evaluation**

```bash
python src/Eval.py
```

---

## Evaluation

Retrieval quality was benchmarked on 5 clinical questions drawn from the ACOG 2018 guideline:

| ID | Topic |
|---|---|
| Q1 | Target fasting and postprandial glucose levels |
| Q2 | Preferred first-line pharmacologic treatment |
| Q3 | Postpartum screening timing and test |
| Q4 | Estimated fetal weight threshold for cesarean counseling |
| Q5 | Exercise recommendations |

**Metrics:** Precision@k, Recall@k, and Hit Rate@k at k = 3 and k = 5. A retrieved chunk counts as relevant when it contains at least half of the question's required keywords.

**Result:** `BGE-Small` achieved the highest hit rate of the models tested (`all-MiniLM-L6-v2`, BGE-Small, and a medical-domain embedding model), with **~73% retrieval accuracy**, and was selected for production.

---

## Safety and limitations

- MedRAG is a **hackathon prototype** and a decision-support aid, **not** a medical device. It must not replace clinical judgment.
- The system prompt instructs the model to answer only from retrieved context, or reply *"I cannot find this in the clinical guidelines."*
- `qwen2.5:0.5b` is a very small model; it can still misread or over-compress retrieved text. Always verify against the source chunks shown in the UI.
- The knowledge base covers a single document (ACOG 2018) and will not reflect newer guidance.

---

## Future work

- Add more maternal-health guidelines
- Clinician feedback loop to improve retrieval
- Upgrade to a larger local model
- Authentication and audit logging for real clinical deployment

---

## Team

Add team name and members here.

## License

Add a license of your choice (e.g., MIT).