============================================
README.md
Ingestion Layer Documentation
=============================

Lovdata RAG System – Ingestion Layer
Production-ready ingestion pipeline for Norwegian legal documents.

The ingestion layer is responsible for **collecting, preprocessing, chunking, embedding, and storing legal documents** into the vector database used by the RAG backend.

============================================
🎯 Features
===========

✅ Lovdata legal document ingestion
✅ Robust preprocessing & text cleaning
✅ Advanced chunking (parent–child, token-aware)
✅ Embedding generation (BGE / OpenAI / Azure OpenAI)
✅ Milvus vector database storage
✅ Idempotent & repeatable ingestion runs
✅ Structured metadata for traceability
✅ Logging & error handling
✅ Production-ready modular architecture

============================================
📁 Project Structure
====================
# Commit
git commit -m "Initial commit - clean codebase with tests"

# Add remote (this will fail if already exists, that's OK)
git remote add origin https://github.com/Vineesh02/Digirett_AI.git

# Force push to overwrite old history
git push -u origin main --force
```

---

### **STEP 11: Verify on GitHub**

1. Go to: https://github.com/Vineesh02/Digirett_AI
2. Refresh the page
3. You should see:
   - ✅ `ingestion/` folder with all your code
   - ✅ `tests/` folder with demo code
   - ✅ `data/` folder (but empty except `.gitkeep`)
   - ✅ Repository size: **< 5 MB**

---
📁 Project Structure
```
DIGIRETT-AI-AGENT/
├── data/
│   ├── .gitkeep
│   └── README.md
├── ingestion/
│   └── src/
│       ├── processors/
│       │   ├── chunker.py
│       │   ├── embedder_sagemaker.py
│       │   └── text_processor.py
│       ├── storage/
│       │   ├── milvus_store.py
│       │   └── supabase_store.py
│       ├── verify/
│       │   ├── check_chunker.py
│       │   ├── del_milvus.py
│       │   ├── verify_milvus.py
│       │   └── verify_sagemaker.py
│       ├── __init__.py
│       ├── config.py
│       └── main.py
├── tests/
│   ├── demo_testing.py
│   ├── test_bge_embedding.py
│   ├── test_collector.py
│   ├── test_health.py
│   ├── test_milvus_store.py
│   └── test_supabase_store.py
├── .gitignore
├── ecosystem.config.js
├── README.md
└── requirements.txt

============================================
🚀 Quick Start
==============

---

1. Install Dependencies

---

```
cd ingestion
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

2. Configure Environment

---

Copy `.env.example` to `.env` and update:

```
# Lovdata
LOVDATA_API_KEY=your-lovdata-api-key

# Embeddings
EMBEDDING_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=bge-m3

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=lovdata_legal_docs

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

---

3. Run the Ingestion Pipeline

---

### Standard Ingestion Run

```
python -m ingestion.src.main
```

This command will:

* Fetch legal documents from Lovdata
* Clean and normalize text
* Apply chunking strategy
* Generate embeddings
* Store vectors and metadata in Milvus

============================================
🧪 Testing
==========

Run ingestion tests:

```
pytest tests/ demo_testing.
```

============================================
🧠 Ingestion Flow
=================

```
Lovdata API
    ↓
Raw Document Loader
    ↓
Text Cleaning & Normalization
    ↓
Chunking (Parent–Child / Token-Aware)
    ↓
Embedding Generation
    ↓
Milvus Vector Store
```

============================================
🧩 Chunking Strategy
====================

* **Parent–Child Chunking**

  * Parent chunk: legal section or article
  * Child chunks: smaller semantic units used for embeddings

* **Token-Aware Chunking**

  * Prevents exceeding LLM token limits
  * Preserves semantic coherence

* **Dynamic Chunk Sizes**

  * Adjusts based on document structure

============================================
📊 Logging & Monitoring
=======================

Logs are stored under `logs/`

```
tail -f logs/ingestion.log
```

Log Levels:

* DEBUG
* INFO
* WARNING
* ERROR

============================================
🔮 Future Enhancements
======================

* Incremental ingestion & versioning
* PDF / DOCX ingestion
* Multilingual embeddings
* Deduplication & change detection
* Ingestion metrics dashboard

============================================
Version: 1.0.0
Last Updated: January 2026
