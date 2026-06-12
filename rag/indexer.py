from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def chunk_document(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(pdf_paths: "str | list[str]", existing_vectorstore=None) -> FAISS:
    """Build or update an in-memory FAISS index from PDF paths."""
    embeddings = _get_embeddings()

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_document(path))

    if existing_vectorstore is not None:
        existing_vectorstore.add_documents(all_chunks)
        return existing_vectorstore

    return FAISS.from_documents(all_chunks, embeddings)
