# External libraries
import streamlit as st
import time

# Internal file imports
from extract import extract_text
from embeddings import create_embeddings

# Start of streamlit application
st.title("PDF QA Bot using Langchain")

## Intitialization
data = ""

st.header("File upload")
file = st.file_uploader("Choose a file (PDF)", type="pdf", help="file to be parsed")

if file is not None:
    with st.spinner("Scraping text from PDF. Please be patient..."):
        time.sleep(5)
    st.success("Done!")

    data = extract_text(file)
    st.text(data, help="Extracted text from uploaded pdf")
    
else:
    st.error("Upload the file to proceed further", icon="🚨")

# Create and display embeddings
if data != "":
    st.header("Create Embeddings")
    embeds = create_embeddings(data)
    st.markdown(embeds)
