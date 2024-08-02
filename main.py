import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub

VECTOR_STORE_PATH = "./vectorstore"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")


# Create a vectorstore from a list of PDFs
def create_vectorstore():
    list_of_pdfs = [
        "pdfs/2024 노무관리 가이드 북.pdf",
    ]

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    documents = []
    for pdf in list_of_pdfs:
        loader = PyPDFLoader(pdf)
        documents += loader.load()

    chunked_documents = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunked_documents, EMBEDDINGS)

    vectorstore.save_local(VECTOR_STORE_PATH)

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    if not os.path.exists(VECTOR_STORE_PATH):
        vectorstore = create_vectorstore()
    else:
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True
        )

    # prompt
    prompt = hub.pull("rlm/rag-prompt")

    # model
    llm = ChatOllama(model="llama3.1:8b")

    # create chain with source
    retriever = vectorstore.as_retriever()
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # query
    query = "연차 계산 방법을 알려주세요."
    response = rag_chain_with_source.invoke(query)

    # print response
    print("Answer:\n", response["answer"] + "\n")
    print("Sources:")
    sources = [doc.metadata for doc in response["context"]]
    for source in sources:
        print(source)


if __name__ == "__main__":
    main()
