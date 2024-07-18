from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyMuPDFLoader
import os

# Load the whole directory certain data type
def load_directory(directory_path, data_type, ocr = False):
    if data_type == "pdf":
        #Use OCR to extract image as text
        if ocr:
            loader_kwargs = {"extract_images":True}
        else:
            loader_kwargs = {"extract_images":False}
        pdf_loader = DirectoryLoader(
            path=directory_path,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader,
            loader_kwargs=loader_kwargs
        )
    pdf_documents = pdf_loader.load()
    return pdf_documents