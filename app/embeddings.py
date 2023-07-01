# Create embeddings of the extracted text for futhrer preprocessing

## Steps involved are:
## - Create chunks of the text so that we can hadle bulk data if any
## - Then create documents of every chunk
## - Finally create embeddings using the Hugging Face embedding class

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

def create_embeddings(text):
    text_splitter = CharacterTextSplitter(        
        separator = "\n\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(openai_api_key="")
    query_result = embeddings.embed_query(texts)
    return query_result
