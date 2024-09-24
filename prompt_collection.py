
from langchain.prompts import ChatPromptTemplate

QA_RAG = "SIMPLE_QUESTION_ANSWER_RAG"
QUESTION_GENERATOR = "QUESTION_GENERATOR"
ANSWER_GENERATOR = "ANSWER_GENERATOR"
EVAL_ANSWER_RELEVANCY = "EVAL_ANSWER_RELEVANCY"
EVAL_FAITHFULNESS = "EVAL_FAITHFULNESS"
GRADING = "GRADING"

prompt_type = {
    "QA_RAG" : "SIMPLE_QUESTION_ANSWER_RAG",
    "QUESTION_GENERATOR" : "QUESTION_GENERATOR",
    "ANSWER_GENERATOR" : "ANSWER_GENERATOR",
    "EVAL_ANSWER_RELEVANCY" : "EVAL_ANSWER_RELEVANCY",
    "EVAL_FAITHFULNESS" : "EVAL_FAITHFULNESS",
    "GRADING" : "GRADING"
}

simple_rag_template = """
Answer the question based on the context below. 
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

bk_question_generation_template = """\
You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

bk_answer_generator_template = """\
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

evaluate_faithfulness_template = """
You are Teaching Assistant. Your task is to evaluate student answer for test question. You are also given Professor's answer as reference. 
Your task is to provide a 'total rating' representing how close student_answer is to the reference.
Give your rating on a scale of 1 to 10, where 1 means that the question is not close at all, and 10 means that the question is extremely close.

Provide your rating as follows:

Total rating: (your rating, as a float number between 1 and 10)

Now here are the question, the student answer and the reference.

Question: {question}

Student Answer: {answer}

Contex: {contexts}

"""

grading_template = """
You are Teaching Assistant. Your task is to extract grade from Professor's comments to student answer. 
You are given some examples of comments for you task. Your answer is ONLY the grade between 1 and 10.

Comment: Total rating: 10.0. The student answer is an exact quote from the reference, which clearly states that Spearman's correlation coefficient was used to calculate the relationship between human-evaluated document relatedness scores and the embedding correlation coefficients for each language model. The student answer matches the reference perfectly, with no deviations or inaccuracies. Therefore, a rating of 10 out of 10 is justified. 
Grade: 10.0

Comment: Total rating: 9.5. The student answer accurately captures the essence of the reference material, correctly interpreting the strong positive correlation of CM_EN as indicating a robust alignment with human judgment in the context of Chinese Medicine. The student also mentions that this implies the model has captured meaningful relationships between documents, which can be used to inform decisions or generate relevant content in the domain of CM.
Grade: 9.5

Comment: Total rating: 8.5
The student answer correctly identifies that a directive is given to the generative model based on GPT-3.5-turbo-16k to minimize hallucination in its response, and mentions the prompt containing this directive. However, it does not accurately cite the specific reference from Document 3, page 7, as mentioned in the student answer. The correct statement is actually found in the Reference material, which states that an alternative prompt without a reference section is passed to a GPT-3.5-turbo-based model to reduce token usage and save on expenses.
Grade: 8.5

Comment: Total rating: 2.0 
The student's answer "I don't know" does not provide any insight into the specific functional limitations of conventional Retrieval-Augmented Generation (RAG) methods for niche domains or how these shortcomings affect their performance. The reference provided, on the other hand, discusses various challenges and considerations associated with RAG systems, including data privacy, scalability, cost, skills required, etc. This suggests a significant gap in understanding between the student's response and the material covered in the lesson.
Grade: 2.0

Comment: Total rating: 4.2
The student answer correctly identifies two of the seven failure points for designing a RAG system (validation during operation and reliance on LLMs). However, they incorrectly infer that these are the only two failure points discussed in the provided snippet, when in fact the reference provides more specific information about the other five failure points. The student's answer also does not fully capture the context of the document and the lessons learned from the case studies. Therefore, while the answer shows some understanding of the topic, it falls short of providing a complete and accurate response.
Grade:4.2

Comment: {comment}
Grade: 

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
    
    if type == prompt_type["EVAL_FAITHFULNESS"]: 
        prompt = ChatPromptTemplate.from_template(evaluate_faithfulness_template)

    if type == prompt_type["GRADING"]: 
        prompt = ChatPromptTemplate.from_template(grading_template)

    return prompt