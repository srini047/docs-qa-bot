# Create a new weaviate vector database and store the result of the generated embeddings

from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader
from pymongo import MongoClient
import streamlit as st

def store_embeddings(docs, embeddings, prompt):
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    uri = st.secrets["MONGODB_CONNECTOR_URL"]
    client = MongoClient(uri)

    db_name = "dev_test_db"
    collection_name = "dev_test_collection"
    collection = client[db_name][collection_name]
    index_name = "dev_test_demo"

    # insert the documents in MongoDB Atlas with their embedding
    vectorStore = MongoDBAtlasVectorSearch(
        collection, embeddings, index_name=index_name
    )
    
    print(vectorStore)

    return vectorStore
