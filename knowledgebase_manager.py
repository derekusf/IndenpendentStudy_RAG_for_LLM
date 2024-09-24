import pandas as pd
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings



CHROMA_OPENAI_RAG_FOR_LLM = "CHROMA_OPENAI_RAG_FOR_LLM"
CHROMA_HF_RAG_FOR_LLM = "CHROMA_HF_RAG_FOR_LLM"
CHROMA_MINILM_RAG_FOR_LLM = "CHROMA_MINILM_RAG_FOR_LLM"
CHROMA_OLLAMA_RAG_FOR_LLM = "CHROMA_OLLAMA_RAG_FOR_LLM"

#IMPORTANT: THE CHROMA INSTANCE CANNOT INITIATED WITHIN A .PY. IT WILL CRASH THE KERNEL. 
class VectorBD:
    
    def __init__(self,
                 vectordb_name) -> None:
        load_dotenv()
#       OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#       print(OPENAI_API_KEY)
        if vectordb_name == CHROMA_OPENAI_RAG_FOR_LLM:
            self.vectordb_directory = os.path.join(os.getenv("VECTORDB_OPENAI_EM"),"RAG_for_LLM")
            self.embeddings = OpenAIEmbeddings()
            self.vectordb =  Chroma(persist_directory=self.vectordb_directory, embedding_function=self.embeddings)
            self.retriever = self.vectordb.as_retriever()

        if vectordb_name == CHROMA_MINILM_RAG_FOR_LLM:
            self.vectordb_directory = os.path.join(os.getenv("VECTORDB_MINILM_EM"),"RAG_for_LLM")
            self.embeddings = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})
            self.vectordb =  Chroma(persist_directory=self.vectordb_directory, embedding_function=self.embeddings)
            self.retriever = self.vectordb.as_retriever()

        if vectordb_name == CHROMA_OLLAMA_RAG_FOR_LLM:
            self.vectordb_directory = os.path.join(os.getenv("VECTORDB_OLLAMA_EM"),"RAG_for_LLM")
            self.embeddings = OllamaEmbeddings(model="llama3.1")
            self.vectordb =  Chroma(persist_directory=self.vectordb_directory, embedding_function=self.embeddings)
            self.retriever = self.vectordb.as_retriever()

        if vectordb_name == CHROMA_HF_RAG_FOR_LLM:
            self.vectordb_directory = os.path.join(os.getenv("VECTORDB_HF_EM"),"RAG_for_LLM")
            self.embeddings = HuggingFaceEmbeddings()
            self.vectordb =  Chroma(persist_directory=self.vectordb_directory, embedding_function=self.embeddings)
            self.retriever = self.vectordb.as_retriever()       

    def vectorizing(self, documents):
        self.vectordb = Chroma.from_documents(documents=documents,embedding=self.embeddings, persist_directory=self.vectordb_directory)
        self.vectordb.persist()

    def invoke(self,question):
#       print(self.retriever.invoke("What is RAG?"))
        return self.retriever.invoke(question)

def connect_km(km_name):
    load_dotenv()
#   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#   print(OPENAI_API_KEY)
    if km_name == CHROMA_OPENAI_RAG_FOR_LLM:
        km_dir = os.path.join(os.getenv("VECTORDB_OPENAI_EM"),"RAG_for_LLM")
        km_embeddings = OpenAIEmbeddings()
        km_db =  Chroma(persist_directory=km_dir, embedding_function=km_embeddings)
        return km_db
    