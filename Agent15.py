from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.vectorstores import Chroma

import prompt_collection3 as myprompt
import llm_connector as myllm
import knowledgebase_manager as km


GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_4 = ""
GPT_4_PREVIEW = ""
LOCAL_GPT4ALL = ""

# Define the agent here
class RAGAgent: 
    rag_type = ""


    def __init__(self,
                 model, vectordb_name, rag_type = "") -> None:

        self.llm = myllm.connectLLM(model)
        self.vectordb = km.VectorBD(vectordb_name) # Chroma(persist_directory=vectordb_directory, embedding_function=embeddings)
        self.retriever = self.vectordb.vectordb.as_retriever()
        if not(rag_type == ""): 
            self.setRAG(rag_type)
    
    def setRAG(self,rag_type):
        if rag_type == myprompt.QA_RAG: 
            setup = RunnableParallel(context=self.retriever, question=RunnablePassthrough())
            prompt =  myprompt.initPrompt(myprompt.QA_RAG) #ChatPromptTemplate.from_template(myprompt.simple_rag_template)
            parser = StrOutputParser()
            self.chain = setup | prompt | self.llm | parser

    def invoke(self,question):
        return self.chain.invoke(question)
    
