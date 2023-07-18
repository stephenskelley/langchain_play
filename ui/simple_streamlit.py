# from https://blog.streamlit.io/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code/
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import sys
sys.path.insert(0, '../')
import local_secrets as secrets

st.title('🦜🔗 Quickstart App')

# openai_api_key = st.sidebar.text_input('OpenAI API Key')
openai_api_key = secrets.techstyle_openai_key
def generate_response(input_text):
  llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key, openai_api_base='http://localhost:1234/v1')
  messages = [
    SystemMessage(
        content="You are a helpful assistant. Please respond to the following request."
    ),
    HumanMessage(
        content=input_text
    ),
]
  st.info(llm(messages).content)

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)