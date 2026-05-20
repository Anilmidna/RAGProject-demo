from pathlib import Path

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

_COLLECTION = "langchain"


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


def _chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def index_exists(persist_dir: str) -> bool:
    if not Path(persist_dir).exists():
        return False
    try:
        col = _chroma_client(persist_dir).get_collection(_COLLECTION)
        return col.count() > 0
    except Exception:
        return False


def clear_index(persist_dir: str = "chroma_db") -> None:
    try:
        _chroma_client(persist_dir).delete_collection(_COLLECTION)
    except Exception:
        pass


def build_index(
    pdf_paths: "str | list[str]",
    persist_dir: str = "chroma_db",
    mode: str = "add",
) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(chunk_document(path))

    if mode == "replace":
        clear_index(persist_dir)
        return Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)

    if index_exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vectorstore.add_documents(all_chunks)
        return vectorstore

    return Chroma.from_documents(all_chunks, embeddings, persist_directory=persist_dir)
