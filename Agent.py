from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Define the agent here
class RAGAgent: 
    
    def __init__(self,
                 llm, embeddings, vectordb) -> None:

        self.llm = llm
        self.embeddings = embeddings
        self.vectordb = vectordb
        self.retriever = vectordb.as_retriever()
        
        setup = RunnableParallel(context=self.retriever, question=RunnablePassthrough())

        template = """
        Answer the question based on the context below. 
        If you can't answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        parser = StrOutputParser()

        self.chain = setup | prompt | llm | parser
        
    def invoke(self,question):
        self.chain.invoke(question)
