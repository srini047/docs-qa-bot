from langchain import text_splitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Weaviate
from langchain import OpenAI
import streamlit as st

def search_qa(data, embeddings, question):
    texts = text_splitter.split_text(data)

    docsearch = Weaviate.from_texts(
        texts,
        embeddings,
        weaviate_url=st.secrets["WEAVIATE_CLUSTER_URL"],
        by_text=False,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
    )
    
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    return chain(
        { "question": str(question) },
        return_only_outputs=True,
    )
    
