import os
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils_embedding import file_loader

load_dotenv()

# Define the directory where the data source files are located
DATASOURCE_DIR = "./datasource_test/"

CONNECTION_STRING = (
    "postgresql+psycopg2://postgres:123456@localhost:5432/medvector_db"
)
COLLECTION_NAME = "doctor_dic"


def process_embedding():
    embeddings = OpenAIEmbeddings()
    loaders = []
    # Iterate over all files in the directory
    for f in os.listdir(DATASOURCE_DIR):
        # If the current item is a file
        file_path = os.path.join(DATASOURCE_DIR, f)
        if os.path.isfile(file_path):
            loaders.append(file_loader(file_path))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    texts = child_splitter.split_documents(docs)

    db = PGVector.from_documents(
        embedding=embeddings,
        documents=texts,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )


if __name__ == "__main__":
    process_embedding()
