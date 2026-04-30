import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print ("Initializing...")

embeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model= "gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0,)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],embedding =embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})    

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on following context only :{context} 
    Question : {question}
    
    Provide a detailed answer:
    """
)
def retrieval_with_lcel():

    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs   
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

def retrieval_chain_without_lcel(query:str):
    """
    Simple retreival chain without LCEL . Manually retrieves documents,formats them and generates a response

    """
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context = context, question= query)
    response = llm.invoke(messages)
    return response.content



def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving ...")

    # Implementation without RAG :
    print("\n"+ "="*70)
    print("1.Implementation without RAG")
    print("="*70)
    query=("what is Pinecone in machine learning?")
    result = llm.invoke([HumanMessage(content= query)])
    print (result.content)

    #Implementation without LCEL :

    print("\n"+ "="*70)
    print("2.Implementation without LCEL")
    print("="*70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\n")
    print(result_without_lcel)

    #Implementation with LCEL :

    print("\n"+ "="*70)
    print("2.Implementation without LCEL")
    print("="*70)
    retrieval_with_lcel = retrieval_with_lcel()
    retrieval_lcel = retrieval_with_lcel.invoke({"question":query})
    print("\n")
    print(retrieval_lcel)




