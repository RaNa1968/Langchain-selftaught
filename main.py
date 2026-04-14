from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
import os

tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """
    tool to search the internet
    Args:
        query: The query to search
    Returns:
        The search result
    """
    print(f"Searching for:",{query})
    return tavily.search(query=query) 

llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")
tools = [search]
agent = create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":[HumanMessage(content= "search for 3 jobs on linkedin in the chennai area for people "
    "with 10+ years of experience")]})
    print(result)


if __name__ == "__main__":
    main()
