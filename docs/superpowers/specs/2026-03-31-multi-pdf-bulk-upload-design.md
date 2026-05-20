# Multi-PDF Bulk Upload — Design Spec

**Date:** 2026-03-31  
**Status:** Approved  
**Scope:** Add a Streamlit web UI for bulk PDF upload and Q&A over a merged ChromaDB index

---

## What We're Building

A single-page Streamlit app (`app.py` at project root) that lets users upload multiple PDFs,
index them into a shared ChromaDB collection, and ask questions against the combined knowledge base.

---

## Architecture

### Changes to `rag/indexer.py`

**1. Extended `build_index()` signature:**

```python
def build_index(
    pdf_paths: str | list[str],
    persist_dir: str = "chroma_db",
    mode: str = "add",   # "add" | "replace"
) -> Chroma
```

- Accepts a single path string or a list of paths (backward compatible)
- `mode="replace"` — deletes `persist_dir` entirely, then indexes all PDFs fresh
- `mode="add"` — if index exists, loads it and calls `add_documents()` with new chunks;
  if no index yet, creates one (same as current cache-first behavior)

**2. New helper `clear_index()`:**

```python
def clear_index(persist_dir: str = "chroma_db") -> None
```

Deletes `persist_dir` and all its contents. Used internally by replace mode and exposed
for the UI's "Clear Index" button.

### No changes to `rag/retriever.py`

`get_qa_chain()` and `get_answer()` are unchanged — they already query whatever is in `persist_dir`.

### New `app.py` at project root

Single-page Streamlit app. No additional files or modules needed.

---

## Streamlit App Layout

### Sidebar — Index Management

| Element | Behavior |
|---------|----------|
| Multi-file uploader | Accepts `.pdf`, multiple files allowed |
| Mode radio | "Add to index" / "Replace index" |
| "Build Index" button | Triggers indexing with a progress spinner |
| Index status | Shows "N chunks indexed" (from `vectorstore._collection.count()`) or "No index yet". File count is tracked only in `st.session_state` for the current session — it resets on page refresh, but the chunk index persists on disk. |
| "Clear Index" button | Clears index after confirmation (`st.warning` + confirm button) |

### Main Area — Q&A

| Element | Behavior |
|---------|----------|
| Text input | "Ask a question about your documents" |
| "Ask" button | Disabled when input is empty |
| Answer box | Styled block with the LLM response |
| "Sources" expander | Shows retrieved chunks with source filename + page number |
| No-index state | Shows "Upload and index PDFs to get started" prompt instead of Q&A |

---

## Data Flow

### Upload & Index

1. User selects PDFs via Streamlit file uploader (receives `UploadedFile` objects with bytes)
2. App writes each file to a `tempfile.TemporaryDirectory` on disk
3. Calls `build_index(pdf_paths=[...], mode="add"|"replace")`
4. On success: shows "Indexed N chunks from X files" — persists result in `st.session_state`
5. Temp directory auto-cleaned when context manager exits

### Q&A

1. User types question → clicks "Ask"
2. App calls `get_answer(question)` → displays `result["answer"]`
3. Sources shown from `result["sources"]` — PyPDFLoader already populates
   `Document.metadata["source"]` (filename) and `Document.metadata["page"]` (page number)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Corrupt/unreadable PDF | Catch exception per file, show `st.warning`, continue with valid files |
| No index when Ask clicked | `st.info("Please upload and index PDFs first")` |
| Empty question | Ask button disabled until text is entered |
| OpenAI API error | Catch exception, show `st.error(...)` inline, don't crash app |

---

## Testing

- Existing `tests/test_indexer.py` tests remain valid — `build_index(single_path)` still works
- New tests for `build_index()`:
  - `test_build_index_accepts_list_of_paths` — passes a list with one PDF, returns Chroma
  - `test_build_index_replace_mode_clears_existing` — verifies old data gone after replace
  - `test_build_index_add_mode_merges_chunks` — verifies chunk count increases after add
  - `test_clear_index_removes_directory` — verifies persist_dir is deleted
- Streamlit UI tested manually (no automated Streamlit tests in this project)

---

## Files Changed

| File | Change |
|------|--------|
| `rag/indexer.py` | Extend `build_index()`, add `clear_index()` |
| `app.py` | New file — Streamlit app |
| `tests/test_indexer.py` | Add 4 new tests for multi-PDF and clear_index |

`rag/retriever.py`, existing tests, `chroma_db/`, `.env` — untouched.
