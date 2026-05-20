
# Multi-PDF Bulk Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the RAG library with a Streamlit web app that lets users bulk-upload PDFs, index them into a shared ChromaDB collection, and ask questions against the combined knowledge base.

**Architecture:** Two changes to `rag/indexer.py` (add `clear_index()`, extend `build_index()` to accept a list and an add/replace mode), then a new `app.py` at the project root that wires Streamlit UI to the existing indexer and retriever. `rag/retriever.py` is untouched.

**Tech Stack:** LangChain, ChromaDB (`langchain-chroma`), OpenAI embeddings + gpt-4o, Streamlit

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rag/indexer.py` | Modify | Add `clear_index()`, extend `build_index()` signature |
| `tests/test_indexer.py` | Modify | Add 4 new tests covering new indexer behaviour |
| `app.py` | Create | Streamlit UI — sidebar for index management, main area for Q&A |

`rag/retriever.py` — **not touched**.

---

## Task 1: Add `clear_index()` to `rag/indexer.py`

**Files:**
- Modify: `rag/indexer.py`
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Write the failing tests**

Add these two tests to the bottom of `tests/test_indexer.py`. Also add `from rag.indexer import clear_index` to the existing import line at the top.

```python
# At top of tests/test_indexer.py, update import:
from rag.indexer import chunk_document, index_exists, clear_index


# Add at the bottom of tests/test_indexer.py:
def test_clear_index_removes_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "chroma_test"
        target.mkdir()
        (target / "test.bin").write_bytes(b"data")
        clear_index(str(target))
        assert not target.exists()


def test_clear_index_is_noop_for_missing_dir():
    # should not raise even if directory does not exist
    clear_index("/nonexistent/path/xyz_does_not_exist_abc")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py::test_clear_index_removes_directory tests/test_indexer.py::test_clear_index_is_noop_for_missing_dir -v
```

Expected: `ImportError: cannot import name 'clear_index'`

- [ ] **Step 3: Implement `clear_index()` in `rag/indexer.py`**

Add `import shutil` at the top and the new function after `index_exists`:

```python
# Add to imports at top of rag/indexer.py:
import shutil


# Add after the index_exists function:
def clear_index(persist_dir: str = "chroma_db") -> None:
    """Delete the ChromaDB persist directory and all its contents."""
    path = Path(persist_dir)
    if path.exists():
        shutil.rmtree(path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py::test_clear_index_removes_directory tests/test_indexer.py::test_clear_index_is_noop_for_missing_dir -v
```

Expected:
```
PASSED tests/test_indexer.py::test_clear_index_removes_directory
PASSED tests/test_indexer.py::test_clear_index_is_noop_for_missing_dir
```

- [ ] **Step 5: Run the full existing test suite to confirm nothing broke**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py -v
```

Expected: all 9 tests pass (7 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/vrln/rag-project
git add rag/indexer.py tests/test_indexer.py
git commit -m "feat: add clear_index() to indexer"
```

---

## Task 2: Extend `build_index()` for multi-PDF and add/replace modes

**Files:**
- Modify: `rag/indexer.py`
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Write the failing tests**

Add these three tests to the bottom of `tests/test_indexer.py`. Also add the mock imports at the top.

```python
# Add to imports at top of tests/test_indexer.py:
from unittest.mock import MagicMock, patch

from rag.indexer import chunk_document, index_exists, clear_index, build_index


# Add at the bottom of tests/test_indexer.py:
def test_build_index_accepts_list_of_paths():
    """build_index() should accept a list of PDF paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist = Path(tmpdir) / "db"
        with patch("rag.indexer.OpenAIEmbeddings"), \
             patch("rag.indexer.Chroma") as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            build_index([PDF_PATH], persist_dir=str(persist), mode="replace")
            assert mock_chroma.from_documents.called


def test_build_index_replace_mode_clears_existing_data():
    """mode='replace' must wipe the persist_dir before re-indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist = Path(tmpdir) / "db"
        persist.mkdir()
        marker = persist / "old_data.bin"
        marker.write_bytes(b"stale data")
        with patch("rag.indexer.OpenAIEmbeddings"), \
             patch("rag.indexer.Chroma") as mock_chroma:
            mock_chroma.from_documents.return_value = MagicMock()
            build_index([PDF_PATH], persist_dir=str(persist), mode="replace")
        assert not marker.exists()


def test_build_index_add_mode_calls_add_documents():
    """mode='add' must call add_documents() on the existing vectorstore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist = Path(tmpdir) / "db"
        persist.mkdir()
        (persist / "existing.bin").write_bytes(b"existing data")
        mock_vectorstore = MagicMock()
        with patch("rag.indexer.OpenAIEmbeddings"), \
             patch("rag.indexer.Chroma") as mock_chroma:
            mock_chroma.return_value = mock_vectorstore
            build_index([PDF_PATH], persist_dir=str(persist), mode="add")
            mock_vectorstore.add_documents.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py::test_build_index_accepts_list_of_paths tests/test_indexer.py::test_build_index_replace_mode_clears_existing_data tests/test_indexer.py::test_build_index_add_mode_calls_add_documents -v
```

Expected: `TypeError` — `build_index()` does not yet accept a list or `mode` parameter.

- [ ] **Step 3: Rewrite `build_index()` in `rag/indexer.py`**

Replace the existing `build_index` function entirely:

```python
def build_index(
    pdf_paths: "str | list[str]",
    persist_dir: str = "chroma_db",
    mode: str = "add",
) -> Chroma:
    """Build or load a ChromaDB index from one or more PDF paths.

    Args:
        pdf_paths: A single PDF path string or a list of PDF path strings.
        persist_dir: Directory where ChromaDB persists data.
        mode: 'replace' clears the existing index before indexing;
              'add' appends new documents to an existing index (creates
              a fresh index if none exists).
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    if mode == "replace":
        clear_index(persist_dir)

    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_document(path))

    if mode == "add" and index_exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vectorstore.add_documents(all_chunks)
        return vectorstore

    return Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)
```

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py::test_build_index_accepts_list_of_paths tests/test_indexer.py::test_build_index_replace_mode_clears_existing_data tests/test_indexer.py::test_build_index_add_mode_calls_add_documents -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run the full test suite to confirm backward compatibility**

```bash
cd /Users/vrln/rag-project
pytest tests/test_indexer.py -v
```

Expected: all 12 tests pass (9 from Task 1 + 3 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/vrln/rag-project
git add rag/indexer.py tests/test_indexer.py
git commit -m "feat: extend build_index() to support multi-PDF and add/replace modes"
```

---

## Task 3: Create Streamlit `app.py`

**Files:**
- Create: `app.py`

- [ ] **Step 1: Install Streamlit**

```bash
pip install streamlit
```

Expected: Streamlit installs successfully. Verify with:

```bash
streamlit --version
```

- [ ] **Step 2: Create `app.py` at the project root**

```python
import tempfile
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag.indexer import build_index, chunk_document, clear_index, index_exists
from rag.retriever import get_answer

PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("PDF Question & Answer")

# ── Sidebar: Index Management ─────────────────────────────────────────────
with st.sidebar:
    st.header("Index Management")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    mode = st.radio(
        "Index mode",
        options=["Add to index", "Replace index"],
    )

    if st.button("Build Index", disabled=not uploaded_files):
        with st.spinner("Indexing…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write each UploadedFile to a real temp path on disk
                all_paths = []
                for f in uploaded_files:
                    tmp_path = Path(tmpdir) / f.name
                    tmp_path.write_bytes(f.read())
                    all_paths.append(str(tmp_path))

                # Validate each PDF; skip corrupt files
                valid_paths, failed = [], []
                for path in all_paths:
                    try:
                        chunk_document(path)
                        valid_paths.append(path)
                    except Exception as exc:
                        failed.append((Path(path).name, str(exc)))

                for name, err in failed:
                    st.warning(f"Skipped '{name}': {err}")

                if valid_paths:
                    index_mode = "replace" if mode == "Replace index" else "add"
                    try:
                        build_index(valid_paths, persist_dir=PERSIST_DIR, mode=index_mode)
                        if index_mode == "replace":
                            st.session_state["session_files"] = []
                        st.session_state.setdefault("session_files", []).extend(
                            Path(p).name for p in valid_paths
                        )
                        st.success(f"Indexed {len(valid_paths)} file(s).")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")

    # Index status
    st.divider()
    if index_exists(PERSIST_DIR):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        chunk_count = vs._collection.count()
        st.success(f"Index ready — {chunk_count} chunks")
        files = st.session_state.get("session_files", [])
        if files:
            st.caption("This session: " + ", ".join(files))
    else:
        st.info("No index yet")

    # Clear index
    st.divider()
    if st.button("Clear Index"):
        st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.warning("Delete all indexed documents?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Confirm", key="confirm_yes"):
                clear_index(PERSIST_DIR)
                st.session_state.pop("confirm_clear", None)
                st.session_state.pop("session_files", None)
                st.rerun()
        with c2:
            if st.button("Cancel", key="confirm_no"):
                st.session_state.pop("confirm_clear", None)
                st.rerun()

# ── Main area: Q&A ────────────────────────────────────────────────────────
if not index_exists(PERSIST_DIR):
    st.info("Upload and index PDFs using the sidebar to get started.")
else:
    question = st.text_input("Ask a question about your documents")

    if st.button("Ask", disabled=not question.strip()):
        with st.spinner("Thinking…"):
            try:
                result = get_answer(question)
                st.markdown("### Answer")
                st.markdown(result["answer"])
                with st.expander("Sources"):
                    for i, chunk in enumerate(result["sources"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.text(chunk)
            except Exception as exc:
                st.error(f"Error: {exc}")
```

- [ ] **Step 3: Run the full test suite once more to confirm app.py imports don't break anything**

```bash
cd /Users/vrln/rag-project
pytest tests/ -v
```

Expected: all 12 tests pass.

- [ ] **Step 4: Start the app and manually verify**

```bash
cd /Users/vrln/rag-project
streamlit run app.py
```

Open `http://localhost:8501` and verify these scenarios:

| Scenario | Expected |
|----------|----------|
| No index on first load | Main area shows "Upload and index PDFs to get started." Sidebar shows "No index yet" |
| Upload 1 PDF, click Build Index (Add mode) | Spinner → "Indexed 1 file(s)." Sidebar status shows chunk count |
| Upload a second PDF, Add mode | Chunk count increases. Both files shown in "This session" caption |
| Upload a PDF, Replace mode | Chunk count resets to only the new file's chunks |
| Ask a question with no text | Ask button remains disabled |
| Ask a valid question | Spinner → answer appears. Sources expander shows retrieved chunks |
| Click Clear Index → Confirm | Index cleared. Main area returns to "Upload and index" prompt |
| Click Clear Index → Cancel | Nothing changes |

- [ ] **Step 5: Commit**

```bash
cd /Users/vrln/rag-project
git add app.py
git commit -m "feat: add Streamlit bulk PDF upload and Q&A app"
```
