
from langchain.prompts import ChatPromptTemplate

simple_rag_template = """
Answer the question based on the context below. 
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

def initPrompt(type):
    if type == "simple_rag": 
        prompt = ChatPromptTemplate.from_template(simple_rag_template)