from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

import streamlit as st

@st.cache_data
def chat_with_pdf(_text, _embeddings, chat_prompt):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(_text)

    docsearch = Weaviate.from_texts(
        texts,
        _embeddings,
        weaviate_url=st.secrets["WEAVIATE_CLUSTER_URL"],
        by_text=False,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    response = chain(
        {"question": chat_prompt},
        return_only_outputs=True,
    )

    print(response["answer"])
    return response["answer"]
