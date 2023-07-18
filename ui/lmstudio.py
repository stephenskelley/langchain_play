from langchain.llms import OpenAIChat
import sys
sys.path.insert(0, './')
import local_secrets as secrets


openai_api_key = secrets.techstyle_openai_key

llm = OpenAIChat(temperature=0.7, openai_api_key=openai_api_key, openai_api_base="http://localhost:1234/v1")
answer = llm('Write an original Shakespearian sonnet')
print(answer)