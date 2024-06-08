import os
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Load environment variables

# Set the Perplexity API key
os.environ["PPLX_API_KEY"] =os.getenv("PPLX_API_KEY")
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an artificial intelligence assistant and you need to "
                   "engage in a helpful, detailed, polite conversation with a user."),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title('Langchain with Perplexity API')
input_text = st.text_input("Ask me anything")

# Perplexity LLM
llm = ChatPerplexity(model="llama-3-sonar-large-32k-online")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))