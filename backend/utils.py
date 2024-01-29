from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.pgvector import PGVector
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

def file_loader(file_path):
    # If the file is a .docx file
    if file_path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path)
    # If the file is a .pdf file
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    # If the file is a .txt file
    elif file_path.endswith(".txt"):
        return TextLoader(file_path)
    # If the file type is not supported
    else:
        # Return None
        return None


def get_relevant_documents(query, embeddings, collection_name, connection_string):
    # Create a PGVector instance
    vector_db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    retriever = vector_db.as_retriever()
    # Return the relevant documents
    return retriever.get_relevant_documents(query=query)


def create_final_retriever(doc_url, embeddings, final_collection_name, final_connection_string):
    final_loader = file_loader(doc_url)
    final_documents = final_loader.load()
    final_text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    final_texts = final_text_splitter.split_documents(final_documents)
    final_db = PGVector.from_documents(
        documents=final_texts,
        embedding=embeddings,
        collection_name=final_collection_name,
        connection_string=final_connection_string,
        pre_delete_collection=True,
    )
    final_retriever = final_db.as_retriever()
    return final_retriever


def get_final_chain(retriever):
    # Initialize the OpenAI model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define the prompt template for the chatbot
    prompt_template = """
    You are an healthcare assistant chatbot. Based on the retrieved context, answer the question. 
    Explain why the information in the context can and, or cannot answer the question. 
    The response is in the following format:
    Recommended doctor is:
    Why it's a good choice:
    What's missing:
    Source is:
    If you don't know the answer, just say that you don't know. 
    Context: {context}
    Question: {question}  
    Answer:
    """

    # Create the prompt template for the chatbot
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create the LangChain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def get_answer(doc_url, query, embeddings, collection_name, connection_string):
    final_retriever = create_final_retriever(
        doc_url, embeddings, collection_name, connection_string
    )
    rag_chain = get_final_chain(final_retriever)
    llm_response = rag_chain.invoke(query)
    return llm_response
