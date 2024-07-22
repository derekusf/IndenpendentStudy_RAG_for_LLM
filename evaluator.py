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

# Call function generate_question and generate_answer to generate a complete testset 
# Output is a dataframe of a complete testset + save it to csv & json files for future use
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

    testset_df = pd.DataFrame(question_ans_context)

    # Write to files for future use
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)
    testset_df.to_csv(os.path.join(testset_path,testfile_csv))
    testset_df.to_json(path_or_buf=os.path.join(testset_path,testfile_json),orient='records',lines=True)
    
    return testset_df

# Generate question by a LLM, for the list of documents used as context
# Per each document used as context, a LLM will compose question about that context
# Output: a Test dataset (lack of answer), each item include a question and a context.  
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
    question_context_list = []

    print(f"evaluator.py log >>> START GENERATING QUESTION")
    i = 1
    for text in tqdm(pdf_documents):
        try:
            response = question_generation_chain.invoke(text.page_content)
        except Exception as e:
            print(f"Exception at {i} {e}")
            i=i+1
            continue
        question_context = {"context": text.page_content, "question" : response}
#        print(f"Question {i} : {question_context["question"]}")
#        print(f"Context {i} : {question_context["context"]}")
        question_context_list.append(question_context)
        i=i+1
    print(f"evaluator.py log >>> COMPLETE GENERATING QUESTION")    
    return question_context_list

# Based on provided questions and contexts, request a LLM to generate answers used as ground truths for later evaluation
# Output: a Test dataset (completed), each item include: a context, a question and an answer aka ground truth
def generate_answer(answer_llm, question_context_list, mode = ""):
    answer = question_context_list
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
    print(f"evaluator.py log >>> START GENERATING ANSWER")
    i = 1
    for record in tqdm(answer):
        try:
            response = answer_generation_chain.invoke({"question":record["question"],"context":record["context"]})
        except Exception as e:
            print(f"Exception at {i} {e}")
            i=i+1
            continue
        record["ground_truth"] = response
        i=i+1
    
    print(f"evaluator.py log >>> COMPLETE GENERATING ANSWER")
    return answer

def rag_evaluate(rag_pipeline):
    ts_path = os.getenv("TS_PROMPT")
    ts_path = os.path.join(ts_path,"RAG_for_LLM","testset_arvix.csv")
    testset_ds = Dataset.from_csv(ts_path)
#    testset_df = pd.DataFrame(testset_ds)
#    testset_df = testset_df.rename(columns={"answer" : "ground_truth"})
#    testset_ds = testset_ds.rename_column("answer","ground_truth")
    test_outcome_list = test_rag_pipeline(rag_pipeline, testset_ds)
    evaluate_llm = myllm.connectLLM("LLAMA3_70B")
    test_outcome_list = evaluate_by_metric(evaluate_llm,test_outcome_list,"answer_relevancy")
    test_outcome_list = evaluate_by_metric(evaluate_llm,test_outcome_list,"faithfulness")

    return test_outcome_list

# Have a LLM to evaluate test outcome to different metrics
# The output is dataset of test outcome + the evaluation value on required metric for each item.
def evaluate_by_metric(critic_llm, test_outcome_list, metric = "answer_relevancy"):
    # How relevant the answer to the question, in the other word, how close the answer to the ground truth
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
        print("evaluator.py log >>> start evaluating answer_relevancy")
        eval_list = []
        for record in tqdm(test_outcome_list):
#            print(f"Question {i} : {record["question"]}")
#            print(f"answer {i} : {record["answer"]}")
#            print(f"ground_truth {i} : {record["ground_truth"]}")
            try:
                response = eval_chain.invoke({"question":record["question"],"answer":record["answer"],"ground_truth":record["ground_truth"]})
            except Exception as e:
                print(f"Exception at {i} {e}")
                i=i+1
                continue
            record["answer_relevancy"] = response
            
#            print(f"answer_relevancy {i} : {record["answer_relevancy"]}")

            """            
            eval_list.append(
                {
                    "question":record["question"],
                       "answer":record["answer"],
                    "ground_truth":record["ground_truth"],
                    "contexts":record["contexts"],
                    "answer_relevancy" : record["answer_relevancy"]
                }
            )"""

            i=i+1
        print("evaluator.py log >>> end evaluating answer_relevancy")
    # How relevant the answer to the question, in the other word, how close the answer to the ground truth
    if metric == "faithfulness": 
        eval_output_parser = StrOutputParser() #StructuredOutputParser.from_response_schemas(answer_response_schemas)
        #setup = RunnableParallel(question = RunnablePassthrough(), context=RunnablePassthrough())

        prompt_template = myprompt.initPrompt(myprompt.EVAL_FAITHFULNESS)

        eval_chain = (
            {"question": itemgetter("question"), "answer": itemgetter("answer"), "contexts": itemgetter("contexts") }
            | prompt_template 
            | critic_llm 
            | eval_output_parser
        )

        i = 1
        print("evaluator.py log >>> start evaluating faithfulness")
        eval_list = []
        for record in tqdm(test_outcome_list):
#            print(f"Question {i} : {record["question"]}")
#            print(f"answer {i} : {record["answer"]}")
#            print(f"ground_truth {i} : {record["ground_truth"]}")
            try:
                response = eval_chain.invoke({"question":record["question"],"answer":record["answer"],"contexts":record["contexts"]})
            except Exception as e:
                print(f"Exception at {i} {e}")
                i=i+1
                continue
            record["faithfulness"] = response
            
#            print(f"faithfulness {i} : {record["faithfulness"]}")
            i=i+1
        print("evaluator.py log >>> start evaluating faithfulness")
    return test_outcome_list # Dataset.from_pandas(pd.DataFrame(eval_list))

# Run testing by letting the RAG pipeline under evaluation answer each question in the testset
# Output is a Test outcome dataset, each item include a question, the answer of rag-pipeline, the contexts that the pipeline retrieved and the ground_truth
def test_rag_pipeline(rag_pipeline, testset_ds):
    i = 1
    test_outcome_list = []
    print(f"evaluator.py log >>> Start testing with on {len(testset_ds)} question")

    for row in tqdm(testset_ds):
        question = row["question"]
        answer = rag_pipeline.invoke(question)
#        print(f"Question {i} : {question} ")
#        print(f"answer {i} : {answer} ")
        test_outcome_list.append(
            {
                "question" : question,
                "answer" : answer,
                "contexts" : [doc.page_content for doc in rag_pipeline.vectordb.invoke(question)],
                "ground_truth" : row["ground_truth"]
            }
        )
        i= i+1
    test_outcome_ds = Dataset.from_pandas(pd.DataFrame(test_outcome_list))
    print(f"evaluator.py log >>> End testing with {len(test_outcome_ds)} answers on {len(testset_ds)} question")
    return test_outcome_list