from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint 
import os
from dotenv import load_dotenv

llm_model = {
    "GPT_3_5_TURBO" : "gpt-3.5-turbo",
    "GPT_4" : "",
    "GPT_4_PREVIEW" : "gpt-4-1106-preview",
    "LOCAL_GPT4ALL" : "",
    "MISRALAI" : "mistralai/Mistral-7B-Instruct-v0.2",
    "LLAMA3_70B" : "meta-llama/Meta-Llama-3-70B-Instruct",
    "ZEPHYR_7B" : "HuggingFaceH4/zephyr-7b-beta"
}

def connectLLM(model):
    load_dotenv()

    # Connect to Open AI chat model: Online, Token-base
    if model == "GPT_3_5_TURBO" or model == "GPT_4_PREVIEW":
#       print("connect llm")
        return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=llm_model[model])
    
    # Connect to HuggingFace chat model: Online, Token-base
    # Note: to use Llama3, we need to register on HuggingFace website
    if model == "LLAMA3_70B":
        repo_id = llm_model[model]
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
         