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
embeddings = OpenAIEmbeddings()

CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:123456@localhost:5432/doctor_vector_db"
)
COLLECTION_NAME = "doctor_lib_vectors"
FINAL_CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:123456@localhost:5432/chosen_vector_db"
)
FINAL_COLLECTION_NAME = "chosen_doctor_vectors"


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


def create_ss_query_chain():
    # Initialize the OpenAI model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define the prompt template for the chatbot
    prompt_template = """
    You are a semantic search assistant. Based on the user's inputs, 
    {question}  
    Rewrite the demande of user in one sentence without subject.
    """

    # Create the prompt template for the chatbot
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    # Create the LangChain
    query_chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return query_chain


def get_relevant_documents(query, collection_name, connection_string):
    # Create a PGVector instance
    vector_db = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
    )
    retriever = vector_db.as_retriever()
    # Return the relevant documents
    return retriever.get_relevant_documents(query=query)


def create_final_retriever(doc_url, final_collection_name, final_connection_string):
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
    The response should complete the following keypoints within the triple backticks:
    ```
    Recommended doctor is:
    Why it's a good choice:
    What's missing:
    Source is:
    ```
    Format the answer as in the example below:
    "
    <p>
    <b>Recommended doctor is:</b> Dr Catherine Lasvergnas Buffet<br>
    <b>Why it's a good choice:</b> Dr Catherine Lasvergnas Buffet offers a full range of orthodontic treatments for children, adolescents, and adults. She is fluent in Portuguese, so she can communicate effectively with Portuguese-speaking patients.<br>
    <b>What's missing:</b> The specific address of Dr Catherine Lasvergnas Buffet's orthodontic practice is not mentioned in the context.<br>
    <b>Source is:</b> ./datasource/Dr Catherine Lasvergnas Buffet Orthodontist.pdf
    </p>
    "
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


def get_answer(doc_url, query, collection_name, connection_string):
    final_retriever = create_final_retriever(
        doc_url, collection_name, connection_string
    )
    rag_chain = get_final_chain(final_retriever)
    llm_response = rag_chain.invoke(query)
    return llm_response


def get_multidoc_rag(query):
    # Optimize symantic search query by LLM
    query_chain = create_ss_query_chain()
    optimized_query = query_chain.invoke({"question": query})

    # Get relevant documents and doc urls
    retrieved_docs = get_relevant_documents(
        optimized_query, COLLECTION_NAME, CONNECTION_STRING
    )

    doc_urls = []
    for doc in retrieved_docs:
        file_url = doc.metadata["source"]
        if file_url not in doc_urls:
            doc_urls.append(file_url)

    # Get answer for each retrieved doc and concatenate them fror the final response
    answers = []
    for doc_url in doc_urls:
        answer = get_answer(
            doc_url, query, FINAL_COLLECTION_NAME, FINAL_CONNECTION_STRING
        )
        answers.append(answer)

    responses_num = [f"{i+1}. {text}" for i, text in enumerate(answers)]
    final_response = "\n\n***************** 💚 **************** 💚 ************** 💚 ********************\n\n".join(responses_num)

    return final_response


def get_rag_answer(query_list, general_chain):
    # # Optimize symantic search query by LLM
    # query_chain = create_ss_query_chain()
    # optimized_query = query_chain.invoke(
    #     {"question": query}
    # )

    query = query_list[-1].lower()

    if query != "yes":
        answer = general_chain.invoke({"question": query})
        return answer
    elif query == "yes":
        last_query = query_list[-2]
        # Get relevant documents and doc urls
        retrieved_docs = get_relevant_documents(last_query, COLLECTION_NAME, CONNECTION_STRING)
        file_url = retrieved_docs[0].metadata["source"]

        # Get answer for each retrieved doc and concatenate them fror the final response
        answer = get_answer(file_url, query, FINAL_COLLECTION_NAME, FINAL_CONNECTION_STRING)

        return answer


def create_general_llm():
    # Initialize the OpenAI model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define the prompt template for the chatbot
    prompt_template = """
    You are a helpful chatbot assistant. Answer  all the user's input, 
    except if the user is looking for healthcare or medical services, ask the user: Would you like me to search in our database?
    Input: {question} 
    Answer: 
    """

    # Create the prompt template for the chatbot
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template,
    )

    # Create the LangChain
    query_chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return query_chain
