from langchain import text_splitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Weaviate
from langchain import OpenAI

def search_qa(data, embeddings, question):
    texts = text_splitter.split_text(data)

    docsearch = Weaviate.from_texts(
        texts,
        embeddings,
        weaviate_url="https://dev-docs-qa-bot-72hr7pq4.weaviate.network",
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
    
