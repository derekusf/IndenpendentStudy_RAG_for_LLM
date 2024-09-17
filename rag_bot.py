import streamlit as st

import Agent

rag1 = Agent.RAGAgent(
    name = "RAG 1 - Simple RAG",
    model = Agent.GPT_3_5_TURBO,
    vectordb_name="CHROMA_OPENAI_RAG_FOR_LLM",
    rag_type= "SIMPLE_QUESTION_ANSWER_RAG"
)

st.set_page_config(page_title="RAG", page_icon=":robot:")
st.header("RAG Bot")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(rag1.invoke(form_input))