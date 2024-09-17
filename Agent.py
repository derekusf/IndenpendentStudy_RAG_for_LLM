from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.vectorstores import Chroma

import prompt_collection as myprompt
import llm_connector as myllm
import knowledgebase_manager as km


GPT_3_5_TURBO = "GPT_3_5_TURBO"
GPT_4 = "GPT_4_PREVIEW"
GPT_4_PREVIEW = "GPT_4_PREVIEW"
LOCAL_GPT4ALL = ""
OLLAMA_LLAMA3 = "OLLAMA_LLAMA3"
OLLAMA_LLAMA3_1 = "OLLAMA_LLAMA3.1"
OLLAMA_GEMMA2 = "OLLAMA_GEMMA2"

# Define the agent here
class RAGAgent: 
    rag_type = ""


    def __init__(self,
                 model, vectordb_name, rag_type = "", name = "") -> None:

        self.llm = myllm.connectLLM(model)
        self.vectordb = km.VectorBD(vectordb_name) # Chroma(persist_directory=vectordb_directory, embedding_function=embeddings)
#       self.retriever = self.vectordb.retriever
        self.rag_type = rag_type
        if not(self.rag_type == ""):
#           print("set rag")
            self.setRAG(rag_type = self.rag_type)
        self.name = name

    def setRAG(self, rag_type = "", name = ""):
        if not(rag_type == ""):
            self.rag_type = rag_type
            if self.rag_type == myprompt.QA_RAG: 
    #           print("set chain")
                self.setup = RunnableParallel(context=self.vectordb.retriever, question=RunnablePassthrough())
                self.prompt =  myprompt.initPrompt(myprompt.QA_RAG) #ChatPromptTemplate.from_template(myprompt.simple_rag_template)
                self.parser = StrOutputParser()
                self.chain = self.setup | self.prompt | self.llm | self.parser
    #           print(self.prompt)
            else:
                print(f"The RAG Type has not been defined")
        if not(name == ""):
            self.name = name

    def invoke(self,question):
        return self.chain.invoke(question)