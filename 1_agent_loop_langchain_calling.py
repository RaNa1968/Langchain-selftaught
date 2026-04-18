##from _typeshed import OpenBinaryMode
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage,SystemMessage,ToolMessage

from langsmith import traceable

Max_iter = 10
Model = "qwen3:1.7b"

@tool
def get_product_price(product:str)->float:
    """Looking up the price of a product in a catalog"""
    print(f"Checking for the price of {product}")
    price ={"laptop":1200.5,"headphones":145.99 , "keyboard": 39.99}
    return price.get(product,0)

@tool
def apply_discount(price:float,tier:str)-> float:
    """Apply a discount tier to a price and return the final price. Available tiers are Gold, Silver and Bronze 
    """
    print(f"Searching for the discount for tier -{tier}")
    discount_pct = {"gold":23,"silver":12,"bronze":5}
    discount = discount_pct.get(tier,0)
    return round(price*(1-discount/100),2) 


# AgentLoop

@traceable(name = "Langchain Agent Loop")
def run_agent(question:str):
    pass


if __name__ =="main" :
    print("Hello to Langchain Agent (.bindtools)!")
    print()
    result = run_agent("What is the price of the laptop after applying a gold discount?")