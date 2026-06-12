import os

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA


def get_qa_chain(persist_dir: str = "chroma_db", api_key: str = None) -> RetrievalQA:
    # API key: passed explicitly (from Streamlit sidebar) or from environment
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=key,
        temperature=0,
        max_tokens=1024,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return chain


def get_answer(question: str, persist_dir: str = "chroma_db", api_key: str = None) -> dict:
    chain = get_qa_chain(persist_dir, api_key=api_key)
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]],
    }
