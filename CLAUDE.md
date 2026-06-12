# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Retrieval-Augmented Generation (RAG)** library for question-answering over PDF documents.

**Pipeline**: PDF → chunked → embedded (`nomic-embed-text` via Ollama) → ChromaDB → retrieved (top-4) → answered by `qwen2.5:7b` via Ollama

**No AWS credentials or API keys required** — runs fully local via Ollama.

## Pre-requisites

Ollama must be running with both models pulled:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b   # students likely have this already
ollama serve              # if not already running
```

> **Important:** If you have an existing `chroma_db/` folder built with the old Bedrock/Titan embeddings (1536-dim), delete it before first run. nomic-embed-text uses 768-dim vectors — the two are incompatible. Use the app's **Clear Index** button or delete the folder manually.

## Commands

```bash
# Install dependencies
pip install langchain langchain-ollama langchain-community langchain-core langchain-chroma pytest

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_indexer.py -v

# Run a single test by name
pytest tests/test_indexer.py::test_chunk_document_returns_list -v
```

## Architecture

Two modules in `rag/`:

### `rag/indexer.py` — Document ingestion
- `chunk_document(pdf_path, chunk_size=1000, chunk_overlap=200)` — loads PDF via PyPDFLoader, splits with RecursiveCharacterTextSplitter
- `index_exists(persist_dir)` — checks if ChromaDB directory already exists and is non-empty
- `build_index(pdf_path, persist_dir="chroma_db")` — **cache-first**: if index already exists it reloads rather than re-embeds

### `rag/retriever.py` — QA chain
- `get_qa_chain(persist_dir="chroma_db")` — wires ChromaDB retriever (k=4) + ChatOpenAI into a RetrievalQA chain
- `get_answer(question, persist_dir="chroma_db")` — returns `{"answer": str, "sources": list[Document]}`

### `chroma_db/` — Runtime vector store
Generated on first `build_index()` call. Delete this directory to force re-indexing.

## Configuration

Requires an `OPENAI_API_KEY` in `.env` at the project root. LangChain reads it automatically via `python-dotenv`.

## Test Design

Tests in `tests/` are unit tests. `test_retriever.py` mocks the QA chain to avoid live API calls. `test_indexer.py` exercises real chunking logic against `Corporate_HR_Policy_Document.pdf` (included in repo).
