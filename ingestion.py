
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
    print("Ingesting...")
    loader = TextLoader("C:/Users/Rahul Naren/source/repos/langchain-course/mediumblog1.txt", encoding= 'UTF-8') ## autodetect_encoding= True also works
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap =0)
    text = text_splitter.split_documents(document)
    print(f"Created {len(text)} chunks")

    # embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview" ,api_key = os.environ.get("GOOGLE_API_KEY"))
    embeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(openai_api_key= os.environ.get("OPENAI_API_KEY"))
    print ("Loading...")

    PineconeVectorStore.from_documents(text, embeddings, index_name = os.environ['INDEX_NAME'])

    print("Finished")

