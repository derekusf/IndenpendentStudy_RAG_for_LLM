import pandas as pd
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

CHROMA_OPENAI_RAG_FOR_LLM = "CHROMA_OPENAI_RAG_FOR_LLM"
class VectorBD:
    
    def __init__(self,
                 vectordb_name) -> None:
        load_dotenv()
        if vectordb_name == CHROMA_OPENAI_RAG_FOR_LLM:
            self.vectordb_directory = os.path.join(os.getenv("VECTORDB_OPENAI_EM"),"RAG_for_LLM")
            self.embeddings = OpenAIEmbeddings()
            self.vectordb =  Chroma(persist_directory=self.vectordb_directory, embedding_function=self.embeddings)
