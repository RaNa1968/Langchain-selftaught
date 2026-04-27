# from _typeshed import OpenBinaryMode
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
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{Model}", temperature = 0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question:{question}")
    print("=" * 60)
    
    messages = [
        SystemMessage(
            content= (
                "you are a helpful shopping assistant"
                "you have a access to product catalog tool and a discount tool.\n\n"
                "Valid products are: laptop, headphones, keyboard. Do NOT use any other value."
                "Strict rules - you MUST follow these steps exactly:\n"
                "1. Never guess the price of the product.You MUST call the get_product_price function to get the real price.\n"
                "2.Only call apply_discount AFTER you have received a price from the get_product_price."
                "Pass the exact price returned by get_product_price into get_discount. Do NOT pass a made up number\n"
                "3.Never calculate discounts yourself using math. Always use the apply_discount tool\n"
                "4.If the user does not provide a tier, ask the user again which tier needs to be used - do NOT assume"
            )
        ),
        HumanMessage(content = question)
    ]

    for iteration in range (1, Max_iter+1):
        print(f"\n--- Iteration-{iteration}---")

        ai_message= llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args {tool_args} ")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not fount")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation),tool_call_id= tool_call_id)
        )

    print("ERROR: max iterations reached without an answer")

    return None

if __name__ == "__main__" :
    print("Hello to Langchain Agent (.bindtools)!")
    print()
    result = run_agent("What is the price of the laptop after applying a gold discount?")