# from langchain import text_splitter
# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Weaviate
# from langchain import OpenAI
import streamlit as st

@st.cache_data
def search_qa(_texts, _embeds, prompt):
    query = prompt
    db = Weaviate.from_documents(_texts, _embeds, weaviate_url=st.secrets["WEAVIATE_CLUSTER_URL"], by_text=False)
    docs = db.similarity_search(query)
    print(docs[0].page_content)
    
    return docs[0].page_content
