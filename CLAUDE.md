# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A **Retrieval-Augmented Generation (RAG)** library for question-answering over PDF documents.

**Pipeline**: PDF → chunked → embedded (OpenAI `text-embedding-3-small`) → ChromaDB → retrieved (top-4) → answered by `gpt-4o`

## Commands

```bash
# Install dependencies
pip install langchain langchain-openai langchain-community langchain-core langchain-chroma pytest

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
