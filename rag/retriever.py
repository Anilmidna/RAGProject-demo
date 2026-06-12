import os

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


def get_answer(question: str, vectorstore: FAISS, api_key: str = None) -> dict:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

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

    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]],
    }
