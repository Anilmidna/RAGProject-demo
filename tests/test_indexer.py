import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.indexer import chunk_document, index_exists, clear_index, build_index

PDF_PATH = "Corporate_HR_Policy_Document.pdf"


def test_chunk_document_returns_list():
    chunks = chunk_document(PDF_PATH)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunk_document_items_have_page_content():
    chunks = chunk_document(PDF_PATH)
    assert all(hasattr(chunk, "page_content") for chunk in chunks)


def test_chunk_document_respects_max_size():
    chunks = chunk_document(PDF_PATH, chunk_size=500)
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)


def test_index_exists_returns_false_for_missing_dir():
    assert index_exists("/nonexistent/path/xyz") is False


def test_index_exists_returns_false_for_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert index_exists(tmpdir) is False


def test_index_exists_returns_true_when_dir_has_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "test.txt").write_text("")
        assert index_exists(tmpdir) is True


def test_chunk_document_raises_for_missing_pdf():
    with pytest.raises(FileNotFoundError):
        chunk_document("/nonexistent/file.pdf")


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
