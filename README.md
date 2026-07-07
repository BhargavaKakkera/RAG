# Conversational PDF RAG Assistant

A modular **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain (LCEL)**, and **ChromaDB** for conversational question answering over one or more PDF documents.

The application indexes uploaded PDFs into a persistent vector database, retrieves relevant document chunks using semantic similarity search, and generates grounded responses with configurable LLM providers. It features conversation memory, document summaries, metadata generation, fingerprint-based embedding reuse, and Dockerized deployment.

---


## Features

*  Multi-PDF upload and indexing
*  Semantic retrieval using ChromaDB
*  Fingerprint-based embedding reuse to avoid unnecessary re-indexing
*  Conversation-aware question answering
*  Automatic document overview, topics, keywords, and table of contents generation
*  Cached document summaries
*  Source citations for every response
*  Configurable LLM providers (Groq, Ollama, Gemini)
*  Configurable embedding providers (HuggingFace, Ollama, Gemini)
*  Configurable chunking and retrieval parameters
*  Dockerized deployment

---

## Architecture

```text
                    User
                      │
                      ▼
              Upload PDF Documents
                      │
                      ▼
              PDF Text Extraction
                      │
                      ▼
                 Text Chunking
                      │
                      ▼
             Embedding Generation
                      │
                      ▼
          ChromaDB Vector Storage
                      │
                      ▼
      Semantic Similarity Retrieval
                      │
          Conversation Memory
                      │
                      ▼
              Selected LLM
                      │
                      ▼
             Grounded Response
```

---

## Tech Stack

| Category         | Technologies                |
| ---------------- | --------------------------- |
| Frontend         | Streamlit                   |
| Framework        | LangChain (LCEL)            |
| Vector Database  | ChromaDB                    |
| Embeddings       | HuggingFace, Ollama, Gemini |
| LLM Providers    | Groq, Ollama, Gemini        |
| PDF Processing   | PyPDF                       |
| Containerization | Docker, Docker Compose      |
| Language         | Python                      |

---

## Project Structure

```text
.
├── app.py
├── config.py
├── chains/
├── embeddings/
├── loaders/
├── memory/
├── prompts/
├── retrievers/
├── utils/
├── vectorstore/
├── chroma_db/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## How It Works

### Document Indexing

1. Upload one or more PDF documents.
2. Extract text from each document.
3. Split the text into semantic chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in ChromaDB.
6. Reuse existing embeddings if the document has already been indexed.

### Question Answering

1. Embed the user query.
2. Retrieve the most relevant document chunks using semantic similarity search.
3. Include conversation history when appropriate.
4. Generate a grounded response using the selected LLM.
5. Return the answer along with supporting document sources.

---

## Design Decisions

### Fingerprint-based Embedding Reuse

Each indexed document is assigned a deterministic fingerprint generated from the uploaded files and chunking configuration. Previously indexed documents reuse their existing Chroma collection, avoiding unnecessary embedding generation.

### Threshold-based Retrieval

Retrieved chunks are filtered using a configurable similarity threshold. If no chunks satisfy the threshold, retrieval automatically relaxes the threshold before falling back to the best available matches.

### Conversation Memory

Session-based conversation history enables follow-up questions while keeping the interaction grounded in previous context.

### Lazy Metadata Generation

Document summaries, topics, keywords, and table of contents are generated only when requested, reducing the initial indexing time.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd conversational-pdf-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GOOGLE_API_KEY=

GROQ_API_KEY=

LLM_PROVIDER=groq

EMBEDDING_PROVIDER=huggingface

CHROMA_PERSIST_DIR=./chroma_db
```

Refer to `config.py` or `.env.example` for all supported configuration options.

---

## Running the Application

```bash
streamlit run app.py
```

---

## Docker

Build the Docker image:

```bash
docker build -t conversational-pdf-rag .
```

Run the container:

```bash
docker run -p 8501:8501 conversational-pdf-rag
```

Or start the application with Docker Compose:

```bash
docker compose up --build
```

The Docker Compose configuration mounts the Chroma persistence directory as a volume so indexed embeddings are preserved across container restarts.

---

## Future Improvements

* DOCX document support
* Improved table extraction
* OCR support for scanned PDFs
* Hybrid retrieval (Vector + BM25)
* Cross-encoder reranking

---
