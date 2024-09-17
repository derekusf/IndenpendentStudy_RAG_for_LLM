from langchain_openai.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint 
from langchain_ollama.chat_models import ChatOllama
import os
from dotenv import load_dotenv

llm_model = {
    "GPT_3_5_TURBO" : "gpt-3.5-turbo",
    "GPT_4" : "",
    "GPT_4_PREVIEW" : "gpt-4-1106-preview",
    "LOCAL_GPT4ALL" : "",
    "MISRALAI" : "mistralai/Mistral-7B-Instruct-v0.2",
    "LLAMA3_70B" : "meta-llama/Meta-Llama-3-70B-Instruct",
    "ZEPHYR_7B" : "HuggingFaceH4/zephyr-7b-beta",
    "OLLAMA_GEMMA2" : "gemma2",
    "OLLAMA_LLAMA3" : "llama3",
    "OLLAMA_LLAMA3.1" : "llama3.1"
}

def connectLLM(model):
    load_dotenv()

    # Connect to Open AI chat model: Online, Token-base
    if model == "GPT_3_5_TURBO" or model == "GPT_4_PREVIEW":
#       print("connect llm")
        return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=llm_model[model])
    
    # Connect to HuggingFace chat model: Online, Token-base
    # Note: to use Llama3, we need to register on HuggingFace website
    if model == "LLAMA3_70B" or model == "MISRALAI" or model == "ZEPHYR_7B":
        repo_id = llm_model[model]
        return HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    
    # Connect to Ollama for Llama3, Llama3.1 and Gemma2 chat models
    # Need these models are working locally, they must have been downloaded. Check instruction for downloading Ollama and models
    if model == "OLLAMA_GEMMA2" or model == "OLLAMA_LLAMA3" or model == "OLLAMA_LLAMA3.1":
        return ChatOllama(model=llm_model[model])
         