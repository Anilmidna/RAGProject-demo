from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA


def get_qa_chain(persist_dir: str = "chroma_db") -> RetrievalQA:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return chain


def get_answer(question: str, persist_dir: str = "chroma_db") -> dict:
    chain = get_qa_chain(persist_dir)
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]],
    }
