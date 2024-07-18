from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm_model = {
    "GPT_3_5_TURBO" : "gpt-3.5-turbo",
    "GPT_4" : "",
    "GPT_4_PREVIEW" : "",
    "LOCAL_GPT4ALL" : ""
}

def connectLLM(model):
    if model == llm_model["GPT_3_5_TURBO"]:
        return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model=model)