
from langchain.prompts import ChatPromptTemplate

QA_RAG = "SIMPLE_QUESTION_ANSWER_RAG"
QUESTION_GENERATOR = "QUESTION_GENERATOR"
ANSWER_GENERATOR = "ANSWER_GENERATOR"
EVAL_ANSWER_RELEVANCY = "EVAL_ANSWER_RELEVANCY"

prompt_type = {
    "QA_RAG" : "SIMPLE_QUESTION_ANSWER_RAG",
    "QUESTION_GENERATOR" : "QUESTION_GENERATOR",
    "ANSWER_GENERATOR" : "ANSWER_GENERATOR",
    "EVAL_ANSWER_RELEVANCY" : "EVAL_ANSWER_RELEVANCY"
}

simple_rag_template = """
Answer the question based on the context below. 
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

question_generation_template_bk = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

answer_generator_template_bk = """\
You are a University Professor creating a test for advanced students. For each question and context, create an answer.

answer: a answer about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""


question_generation_template = """
You are a University Professor creating a test for advanced students. 
Based on the given context, create a open question that is specific to the context. 
Your question should be answerable with a specific, concise piece of factual information from the context.
Your question is not multiple choice question. 
Your question should be formulated in the same style as questions users could ask in a search engine.
This means that your question MUST NOT mention something like "according to the passage" or "context".


Provide your anwser as follows: 

Question: (your question)

Here is the context.

context: {context}
"""

answer_generator_template = """
You are Teaching Assistant. Your task is to answer the question based on the context below. 
Your answer should be specific, based on concise piece of factual information from the context. 
Your answer MUST NOT mention something like "according to the passage".
If you can't answer the question, reply "I don't know".

Provide your anwser as follows: 

Answer: (your answer)

Here are the question and context

question: {question}
context: {context}
"""

evaluate_answer_relevancy_template = """
You are Teaching Assistant. Your task is to evaluate student_answer for test_question. You are also given Professor's answer as reference. 
Your task is to provide a 'total rating' representing how close student_answer is to the reference.
Give your rating on a scale of 1 to 10, where 1 means that the question is not close at all, and 10 means that the question is extremely close.

Provide your rating as follows:

Total rating: (your rating, as a float number between 1 and 10)

Now here are the question, the student answer and the reference.

Question: {question}

Student Answer: {answer}

Reference: {ground_truth}

"""

def initPrompt(type) -> ChatPromptTemplate:
    #default
    prompt = ChatPromptTemplate.from_template(simple_rag_template)
    if type == prompt_type["QA_RAG"]: 
        prompt = ChatPromptTemplate.from_template(simple_rag_template)

    if type == prompt_type["QUESTION_GENERATOR"]: 
        prompt = ChatPromptTemplate.from_template(question_generation_template) 

    if type == prompt_type["ANSWER_GENERATOR"]: 
        prompt = ChatPromptTemplate.from_template(answer_generator_template) 

    if type == prompt_type["EVAL_ANSWER_RELEVANCY"]: 
        prompt = ChatPromptTemplate.from_template(evaluate_answer_relevancy_template) 
    return prompt