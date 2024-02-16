from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader


def file_loader(file_path):
    # If the file is a .docx file
    if file_path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(file_path)
    # If the file is a .pdf file
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    # If the file is a .txt file
    elif file_path.endswith(".txt"):
        return TextLoader(file_path, encoding='utf-8')
    # If the file type is not supported
    else:
        # Return None
        return None
