from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatPerplexity
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()


os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
## Langmith tracking

app = FastAPI(
    title="LangChain API",
    version="1.0",
    description="Simple API server"
)
add_routes(
    app,
    ChatPerplexity(model="llama-3-sonar-large-32k-online"),
    path="/perplexity"
 )
model=ChatPerplexity(model="llama-3-sonar-large-32k-online")
llm=Ollama(model="llama3")

prompt1=ChatPromptTemplate.from_template("{topic}")
prompt2=ChatPromptTemplate.from_template("{topic}")

add_routes(
    app,
    prompt1|model,
    path="/Online_model"
)

add_routes(
    app,
    prompt2|llm,
    path="/Offline_model"


)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)

