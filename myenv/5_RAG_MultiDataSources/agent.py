from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()
os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Wikipedia API setup
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Document loader and vector store setup
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OllamaEmbeddings(model="llama3"))
retriever = vectordb.as_retriever()

# Custom LangSmith retriever tool
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Arxiv API setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Sequence of tools
tools = [wiki, arxiv, retriever_tool]

# Perplexity LLM setup
llm = ChatPerplexity(model="llama-3-sonar-large-32k-online")

# Get the prompt to use from lots of templates available
prompt = hub.pull("hwchase17/openai-functions-agent")


# Create the StrOutputParser instance
output_parser = StrOutputParser()

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor and add the StrOutputParser to the chain
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor | output_parser

# Invoke the chain with the input
result = chain.invoke({'input': "What is Dieter Schwarz Stiftung?"})
print(result)