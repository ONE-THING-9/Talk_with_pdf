import streamlit as st
import google.generativeai as genai
from rag import RAG
import PyPDF2
genai.configure(api_key="AIzaSyDNHd-Pcbk8zOfwNlr91rc1aG00ci9wIqQ")
model = genai.GenerativeModel('gemini-pro')
rrag = RAG()
with st.sidebar:
    api_key = st.text_input("API KEY", key="file_qa_api_key", type="password")

st.title("📝 File Q&A with Exprt")
uploaded_file = st.file_uploader("Upload an article", type=("pdf"))
if uploaded_file:
    print(uploaded_file)
    rrag.parse_file(uploaded_file)

question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)
if uploaded_file and question and not api_key:
    st.info("Please add your Anthropic API key to continue.")

if uploaded_file and question and api_key:
    context = rrag.retrieve(question)
    prompt = f"question"
    response = model.generate_content(context + '\n' + question)

    print(response)

    st.write("### Answer")
    st.write(str(response.parts[0])[5:])
    st.write("### context")
    st.write(context)