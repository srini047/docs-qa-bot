# from langchain import text_splitter
# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Weaviate
# from langchain import OpenAI
import streamlit as st
# from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

def search_qa(texts, embeds, prompt):
    db = Weaviate.from_documents(texts, embeds, weaviate_url=st.secrets["WEAVIATE_CLUSTER_URL"], by_text=False)

    docs = db.similarity_search(prompt)

    return docs[0]
