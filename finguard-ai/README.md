# FinGuard AI — Phase 1

> **AI-Powered Financial Compliance & Fraud Investigation Assistant**
> RAG system built with FastAPI · LangChain · Gemini 1.5 Flash · ChromaDB

---

## Features

| Feature | Detail |
|---|---|
| PDF Upload | Multipart upload with size & type validation |
| Document Processing | PyPDFLoader → RecursiveCharacterTextSplitter |
| Embeddings | Google `embedding-001` via LangChain |
| Vector Store | Persistent ChromaDB (local disk) |
| LLM | Gemini 1.5 Flash (grounded, source-cited answers) |
| API | FastAPI with Pydantic v2 schemas, OpenAPI docs |

---

## Project Structure

```
finguard-ai/
├── app/
│   ├── main.py                   # FastAPI app factory + lifespan
│   ├── core/
│   │   ├── config.py             # pydantic-settings (all env vars)
│   │   └── exceptions.py         # Domain exceptions
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   ├── rag/
│   │   ├── document_processor.py # PDF load + chunk + metadata
│   │   ├── vector_store.py       # ChromaDB singleton wrapper
│   │   └── rag_chain.py          # Retriever + prompt + Gemini chain
│   ├── routes/
│   │   ├── health.py             # GET  /health
│   │   ├── upload.py             # POST /api/v1/upload
│   │   └── ask.py                # POST /api/v1/ask
│   ├── services/
│   │   ├── upload_service.py     # Orchestrates ingestion
│   │   └── query_service.py      # Orchestrates RAG query
│   └── utils/
├── data/
│   └── uploads/                  # Saved PDF files
├── chroma_db/                    # Persistent ChromaDB files
├── requirements.txt
├── .env                          # Environment variables (do not commit)
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A **Google AI Studio** API key → [Get one here](https://aistudio.google.com/app/apikey)

### 2. Clone and set up virtual environment

```bash
git clone <repo-url>
cd finguard-ai

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env .env.local                # or just edit .env directly
```

Open `.env` and set:

```env
GOOGLE_API_KEY=your_actual_key_here
```

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

Server starts at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

## API Reference

### `GET /health`

Liveness probe. Returns app status and number of indexed chunks.

```json
{
  "status": "ok",
  "app": "FinGuard AI",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### `POST /api/v1/upload`

Upload and index a PDF document.

**Request:** `multipart/form-data`
- `file` — PDF file (max 50 MB by default)

**Response `201`:**
```json
{
  "message": "PDF successfully processed and indexed.",
  "filename": "AML_Policy.pdf",
  "chunks_indexed": 47,
  "collection": "finguard_documents"
}
```

**Error responses:**

| Code | Reason |
|------|--------|
| 413 | File exceeds size limit |
| 415 | Not a PDF file |
| 422 | PDF contains no extractable text |

---

### `POST /api/v1/ask`

Ask a compliance or fraud investigation question.

**Request body:**
```json
{
  "question": "What AML rule applies when transaction thresholds are exceeded?"
}
```

**Response `200`:**
```json
{
  "question": "What AML rule applies when transaction thresholds are exceeded?",
  "answer": "According to the AML Policy, transactions exceeding $10,000 must be reported via a Currency Transaction Report (CTR) within 15 days...",
  "sources": [
    {
      "filename": "AML_Policy.pdf",
      "page": 4,
      "chunk_index": 12,
      "excerpt": "Section 3.2 — Currency Transaction Reporting: All cash transactions..."
    }
  ],
  "model_used": "gemini-1.5-flash"
}
```

**Error responses:**

| Code | Reason |
|------|--------|
| 400 | Empty question |
| 409 | No documents indexed yet |
| 502 | Gemini or ChromaDB service error |

---

## Example cURL Calls

```bash
# Health check
curl http://localhost:8000/health

# Upload a PDF
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@/path/to/AML_Policy.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What AML rule applies when transaction thresholds are exceeded?"}'
```

---

## Configuration Reference

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | _(required)_ | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model to use |
| `GEMINI_TEMPERATURE` | `0.2` | LLM temperature (lower = more factual) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `5` | Chunks to retrieve per query |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF file size |
| `CHROMA_PERSIST_DIR` | `chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | `data/uploads` | Saved PDF storage path |

---

## Architecture Notes

```
POST /upload
  └── UploadService.ingest()
        ├── validate file type / size
        ├── save to data/uploads/
        ├── document_processor.load_and_chunk_pdf()
        │     ├── PyPDFLoader → pages
        │     └── RecursiveCharacterTextSplitter → chunks + metadata
        └── VectorStore.add_documents()
              └── GoogleGenerativeAIEmbeddings → ChromaDB

POST /ask
  └── QueryService.answer()
        └── RAGChain.run()
              ├── VectorStore.as_retriever().invoke(question)
              ├── format context from retrieved chunks
              ├── ChatPromptTemplate → ChatGoogleGenerativeAI
              └── return AskResponse { answer, sources, model_used }
```
