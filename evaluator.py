from langchain.output_parsers import ResponseSchema
#from langchain.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
#from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from tqdm import tqdm
import os
from dotenv import load_dotenv
import pandas as pd
from operator import itemgetter
from datasets import Dataset

import prompt_collection as myprompt
import document_handler as dc
import llm_connector as myllm

ARVIX_RAG_FOR_LLM = "ARVIX_RAG_FOR_LLM"

def generate_testset(source_doc):
    load_dotenv()
    question_ans_context = []

    if source_doc == ARVIX_RAG_FOR_LLM:
        directory_path = os.path.join(os.getenv("DOC_ARVIX"),"RAG_for_LLM") 
        testset_path = os.path.join(os.getenv("TS_PROMPT"),"RAG_for_LLM")
        testfile_csv = "testset_arvix.csv"
        testfile_json = "testset_arvix.json"
    else: 
        directory_path = "./"
        testset_path = "./"
        testfile_csv = "testset.csv"
        testfile_json = "testset.json"
    pdf_documents = dc.load_directory(directory_path,"pdf")

    generator_llm = myllm.connectLLM("LLAMA3_70B")

    question_context = generate_question(generator_llm, pdf_documents)

    answer_llm = myllm.connectLLM("LLAMA3_70B")

    question_ans_context = generate_answer(answer_llm,question_context)

    ts = pd.DataFrame(question_ans_context)

    # Write to files for future use
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)
    ts.to_csv(os.path.join(testset_path,testfile_csv))
    ts.to_json(path_or_buf=os.path.join(testset_path,testfile_json),orient='records',lines=True)
    
    return ts

# Generate question by a LLM, for the list of documents used as context
def generate_question(generator_llm, pdf_documents, mode = ""):

    
    question_schema = ResponseSchema(
        name="question",
        description="a question about the context."
    )
    question_response_schemas = [
        question_schema,
    ]
    question_output_parser =  StrOutputParser() #StructuredOutputParser.from_response_schemas(question_response_schemas)
    prompt_template = myprompt.initPrompt(myprompt.QUESTION_GENERATOR)
    setup = RunnableParallel(context=RunnablePassthrough())
    question_generation_chain = setup | prompt_template | generator_llm | question_output_parser
    question_ans_context = []
 
    i = 1
    for text in tqdm(pdf_documents):
        try:
            response = question_generation_chain.invoke(text.page_content)
        except Exception as e:
            print(f"Exception at {i} {e}")
            i=i+1
            continue
        qa = {"context": text.page_content, "question" : response}
        print(f"Question {i} : {qa["question"]}")
        print(f"Context {i} : {qa["context"]}")
        question_ans_context.append(qa)
        i=i+1
    
    return question_ans_context

#Based on provided questions and contexts, request a LLM to generate answers used as ground truths for later evaluation
def generate_answer(answer_llm, questions, mode = ""):
    answer = questions
    answer_schema = ResponseSchema(
        name="answer",
        description="an answer to the question"
    )
    answer_response_schemas = [
        answer_schema,
    ]
    answer_output_parser = StrOutputParser() #StructuredOutputParser.from_response_schemas(answer_response_schemas)
    #setup = RunnableParallel(question = RunnablePassthrough(), context=RunnablePassthrough())

    prompt_template = myprompt.initPrompt(myprompt.ANSWER_GENERATOR)

    answer_generation_chain = (
        {"question": itemgetter("question"), "context": itemgetter("context") }
        | prompt_template 
        | answer_llm 
        | answer_output_parser
    )

    i = 1
    for record in tqdm(answer):
        try:
            response = answer_generation_chain.invoke({"question":record["question"],"context":record["context"]})
        except Exception as e:
            print(f"Exception at {i} {e}")
            i=i+1
            continue
        record["answer"] = response
        i=i+1
    return answer

def rag_evaluate(rag_pipeline):
    ts_path = os.getenv("TS_PROMPT")
    ts_path = os.path.join(ts_path,"RAG_for_LLM","testset_arvix.csv")
    testset_ds = Dataset.from_csv(ts_path)
    testset_df = pd.DataFrame(testset_ds)
    testset_df = testset_df.rename(columns={"answer" : "ground_truth"})
    testset_ds = testset_ds.rename_column("answer","ground_truth")
    rag_eval_ds = create_eval_dataset(rag_pipeline, testset_ds)
    answer_llm = myllm.connectLLM("LLAMA3_70B")
    return evaluate(answer_llm,rag_eval_ds) 

def evaluate(critic_llm, eval_ds, metric = "answer_relevancy"):
    if metric == "answer_relevancy": 
        eval_output_parser = StrOutputParser() #StructuredOutputParser.from_response_schemas(answer_response_schemas)
        #setup = RunnableParallel(question = RunnablePassthrough(), context=RunnablePassthrough())

        prompt_template = myprompt.initPrompt(myprompt.EVAL_ANSWER_RELEVANCY)

        eval_chain = (
            {"question": itemgetter("question"), "answer": itemgetter("answer"), "ground_truth": itemgetter("ground_truth") }
            | prompt_template 
            | critic_llm 
            | eval_output_parser
        )

        i = 1
        print("start evaluating")
        eval_list = []
        for record in tqdm(eval_ds):
            print(f"Question {i} : {record["question"]}")
            print(f"answer {i} : {record["answer"]}")
            print(f"ground_truth {i} : {record["ground_truth"]}")
            try:
                response = eval_chain.invoke({"question":record["question"],"answer":record["answer"],"ground_truth":record["ground_truth"]})
            except Exception as e:
                print(f"Exception at {i} {e}")
                i=i+1
                continue
            record["answer_relevancy"] = response
            print(f"answer_relevancy {i} : {record["answer_relevancy"]}")
            eval_list.append(
                {
                    "question":record["question"],
                    "answer":record["answer"],
                    "ground_truth":record["ground_truth"],
                    "contexts":record["contexts"],
                    "answer_relevancy" : record["answer_relevancy"]
                }
            )
            i=i+1
        return Dataset.from_pandas(pd.DataFrame(eval_list))


def create_eval_dataset(rag_pipeline, testset_ds):
    i = 1
    rag_dataset = []
    for row in tqdm(testset_ds):
        question = row["question"]
        answer = rag_pipeline.invoke(question)
        print(f"Question {i} : {question} ")
        print(f"answer {i} : {answer} ")
        rag_dataset.append(
            {
                "question" : question,
                "answer" : answer,
                "contexts" : [doc.page_content for doc in rag_pipeline.vectordb.invoke(question)],
                "ground_truth" : row["ground_truth"]
            }
        )
        i= i+1
    print(f"End creating rag_dataset {len(rag_dataset)}")
    rag_eval_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_eval_df)
    print(f"End creating eval ds {len(rag_eval_dataset)}")
    return rag_eval_dataset