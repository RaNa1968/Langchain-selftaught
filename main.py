from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    tool to search the internet
    Args:
        query: The query to search
    Returns:
        The search result
    """
    print(f"Searching for:", {query})
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for the source used by the agent"""

    url: str = Field(description="Source URL")


class AgentResponse(BaseModel):
    """Schema for the agent response with URL"""

    answer: str = Field(description="The returend result from the search")
    answer_link: List[Source] = Field(
        default_factory=list, description="The url of the returned answer"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0,
)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 jobs on linkedin in the chennai area for employees with 10+ years of experience in the AI/ML domain"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
