# RAG PDF Q&A

A local Retrieval-Augmented Generation (RAG) app for question-answering over PDF documents. Upload PDFs, build a vector index, and ask natural-language questions — answers are grounded in the actual content of your documents.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  INDEXING                                               │
│  PDF Files → Chunking → OpenAI Embedding → ChromaDB    │
├─────────────────────────────────────────────────────────┤
│  QUERYING                                               │
│  Question → Embed → ChromaDB Search → Top-4 Chunks     │
│          → GPT-4o → Grounded Answer                     │
└─────────────────────────────────────────────────────────┘
```

**Indexing:** Each PDF is split into 1000-character chunks with 200-character overlap, embedded using OpenAI `text-embedding-3-small`, and stored in a local ChromaDB vector database.

**Querying:** The question is embedded with the same model, ChromaDB finds the 4 most similar chunks by cosine similarity, and GPT-4o generates an answer grounded in those chunks.

## Project Structure

```
rag-project/
├── app.py                        # Streamlit UI
├── rag/
│   ├── indexer.py                # PDF loading, chunking, ChromaDB index management
│   └── retriever.py              # RetrievalQA chain (ChromaDB + GPT-4o)
├── tests/
│   ├── test_indexer.py
│   └── test_retriever.py
├── chroma_db/                    # Vector store (auto-created, gitignored)
└── Corporate_HR_Policy_Document.pdf   # Sample PDF for testing
```

## Setup

**Prerequisites:** Python 3.11+, an OpenAI API key.

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install streamlit langchain langchain-openai langchain-community \
            langchain-core langchain-chroma langchain-classic \
            langchain-text-splitters pypdf python-dotenv pytest

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Start the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

## Usage

1. **Upload PDFs** — drag one or more PDFs into the sidebar uploader.
2. **Choose index mode:**
   - *Add to index* — appends new documents to the existing store.
   - *Replace index* — wipes the current store and rebuilds from scratch.
3. **Click Build Index** — chunks and embeds the PDFs into ChromaDB.
4. **Ask a question** — type a question in the main area and click Ask.
5. **View sources** — expand the Sources section to see the exact chunks retrieved from ChromaDB.

## Configuration

| Setting | Default | Where |
|---------|---------|-------|
| Embedding model | `text-embedding-3-small` | `rag/indexer.py`, `rag/retriever.py` |
| LLM | `gpt-4o` | `rag/retriever.py` |
| Chunk size | 1000 chars | `rag/indexer.py` |
| Chunk overlap | 200 chars | `rag/indexer.py` |
| Top-k retrieval | 4 chunks | `rag/retriever.py` |
| Vector store path | `chroma_db/` | `app.py` (`PERSIST_DIR`) |

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

`test_indexer.py` runs real chunking against the included sample PDF.  
`test_retriever.py` mocks the QA chain to avoid live API calls.

## Key Design Notes

- **ChromaDB is never deleted** — the `clear_index` and `replace` operations use ChromaDB's own collection API (`delete_collection`) rather than deleting the SQLite file on disk. Deleting the file while the chromadb process singleton holds it open causes `SQLITE_READONLY_DBMOVED` errors.
- **`index_exists` checks collection state** — it queries the actual ChromaDB collection count, not just whether the `chroma_db/` directory exists, so it correctly returns `False` after a clear.
- **Virtual environment required** — the Streamlit binary must come from the venv (system Python versions differ). Always `source venv/bin/activate` before running.
