# FinGuard AI — Phase 1

AI-Powered Financial Compliance & Fraud Investigation Assistant built using FastAPI, LangChain, ChromaDB, and Google Gemini.

## Overview

FinGuard AI is a Retrieval-Augmented Generation (RAG) system designed to help users analyze financial compliance and fraud-related documents through natural language queries.

The system ingests PDF documents, converts them into vector embeddings, stores them in ChromaDB, retrieves contextually relevant information using semantic search, and leverages Google Gemini to generate grounded responses.

---

## Current Status

### Phase 1 Completed

* PDF Upload API
* Document Processing Pipeline
* Text Chunking & Metadata Extraction
* ChromaDB Vector Storage Integration
* FastAPI REST APIs
* RAG Architecture Implementation
* Semantic Retrieval Pipeline
* Google Gemini Integration

### Planned Enhancements

* User Authentication & Authorization
* Multi-Document Analysis
* Conversation Memory
* Fraud Risk Scoring
* Compliance Report Generation
* Docker Deployment
* Cloud Deployment

---

## Features

| Feature              | Description                                    |
| -------------------- | ---------------------------------------------- |
| PDF Upload           | Upload compliance and financial documents      |
| Document Processing  | PDF parsing, chunking, and metadata extraction |
| Semantic Search      | Vector similarity search using embeddings      |
| Vector Database      | Persistent ChromaDB storage                    |
| LLM Integration      | Google Gemini-powered response generation      |
| REST APIs            | FastAPI endpoints with OpenAPI documentation   |
| Modular Architecture | Layered service-oriented backend design        |

---

## Tech Stack

### Backend

* Python
* FastAPI
* Pydantic v2

### AI & NLP

* LangChain
* Google Gemini
* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Semantic Search

### Vector Database

* ChromaDB

### Document Processing

* PyPDF
* RecursiveCharacterTextSplitter

### Infrastructure

* Git
* GitHub

---

## Project Structure

```text
finguard-ai/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── models/
│   │   └── schemas.py
│   ├── rag/
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   └── rag_chain.py
│   ├── routes/
│   │   ├── health.py
│   │   ├── upload.py
│   │   └── ask.py
│   ├── services/
│   │   ├── upload_service.py
│   │   └── query_service.py
│   └── utils/
├── data/
├── chroma_db/
├── requirements.txt
├── .env
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/amankhan1310/finguard-ai.git
cd finguard-ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here

GEMINI_MODEL=gemini-1.5-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_TOP_K=5

MAX_UPLOAD_SIZE_MB=50

CHROMA_PERSIST_DIR=chroma_db
UPLOAD_DIR=data/uploads
```

---

## Running the Application

```bash
uvicorn app.main:app --reload
```

Server:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns application status and health information.

---

### Upload Document

```http
POST /api/v1/upload
```

Uploads and indexes PDF documents into the vector database.

---

### Ask Questions

```http
POST /api/v1/ask
```

Submit natural language questions related to uploaded documents.

Example:

```json
{
  "question": "What AML rules apply when transaction thresholds are exceeded?"
}
```

---

## Architecture

```text
PDF Upload
    ↓
Document Processing
    ↓
Chunking
    ↓
Vector Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Google Gemini
    ↓
Grounded Response
```

---

## Key Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Large Language Model Integration
* Vector Databases
* Semantic Search
* FastAPI Backend Development
* Modular Software Architecture
* REST API Design
* Document Intelligence Systems
* NLP Application Development

---

## Author

Aman Yahya Khan

LinkedIn:
https://linkedin.com/in/aman-khan-40804a283

GitHub:
https://github.com/amankhan1310
