
from langchain.prompts import ChatPromptTemplate

simple_rag_template = """
Answer the question based on the context below. 
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

question_generation_template = """
Answer the question based on the context below. 
If you can't answer the question, reply "I don't know".

Context: {context}
"""

def initPrompt(type):
    if type == "simple_rag": 
        prompt = ChatPromptTemplate.from_template(simple_rag_template)

    if type == "question_generator": 
        prompt = ChatPromptTemplate.from_template(question_generation_template)