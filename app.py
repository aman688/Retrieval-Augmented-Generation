import streamlit as st
import pdfplumber
import time
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv(r'C:\Users\amang\OneDrive\Desktop\project\.env')

# Initialize Pinecone and OpenAI API keys
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set up Pinecone index
pc = Pinecone(api_key=pinecone_api_key)
index_name = "any_index_name"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Wait for index to be ready
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

# Set up OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

# Set up Streamlit app
st.title("any_index_name Guide")
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

if pdf_file:
    # Read PDF file
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Split text into sections
    headers_to_split_on = [("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(text)

    # Create Pinecone vector store
    namespace = "any_index_name"
    docsearch = PineconeVectorStore.from_documents(
        documents=md_header_splits,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    # Set up RetrievalQA chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4-1106-preview", temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    # Get user query
    query = st.text_input("Ask a question about the any_index_name")

    # Answer question
    if query:
        answer = qa.invoke(query)
        st.write(answer)
