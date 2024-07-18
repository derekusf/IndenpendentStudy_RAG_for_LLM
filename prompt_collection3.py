
from langchain.prompts import ChatPromptTemplate

QA_RAG = "SIMPLE_QUESTION_ANSWER_RAG",
QUESTION_GENERATOR = "QUESTION_GENERATOR"

prompt_type = {
    "QA_RAG" : "SIMPLE_QUESTION_ANSWER_RAG",
    "QUESTION_GENERATOR" : "QUESTION_GENERATOR"
}

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

def initPrompt(type) -> ChatPromptTemplate:
    #default
    prompt = ChatPromptTemplate.from_template(simple_rag_template)
    if type == prompt_type["QA_RAG"]: 
        prompt = ChatPromptTemplate.from_template(simple_rag_template)

    if type == prompt_type["QUESTION_GENERATOR"]: 
        prompt = ChatPromptTemplate.from_template(question_generation_template) 
    return prompt