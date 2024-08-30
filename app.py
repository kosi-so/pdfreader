# Import Libraries 
import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import OpenAI
import os 
import streamlit as st 
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()

## Initialise Streamlit App
st.set_page_config(page_title="PDF Reader")
st.header("PDF Reader")


# Read Input PDF 
def read_doc (directory):
    file_loader = PyPDFDirectoryLoader(directory)
    document = file_loader.load()
    return document

doc = read_doc("documents/")

## Divide documents into the chunks 
def chunk_data (docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc =text_splitter.split_documents(docs)
    return docs

doc_chunk = chunk_data(docs=doc)

## Embed Document Chunks
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

## Initialise Pinecone database
pinecone_api_key = os.environ["PINECONE_API_KEY"]

vector_db = Pinecone.from_documents(
        documents = doc_chunk, 
        embedding = OpenAIEmbeddings(), 
        index_name = "llm-app",)

## Retrieve result with cosine similarity 
def retrieve_query(query, k=4):
    matching_result = vector_db.similarity_search(query, k=k)
    return matching_result


# INitialising LLM model 
llm =OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.6)
chain = load_qa_chain(llm, chain_type='stuff')

# Retrieving answers 
def retrieve_answers(query):
    doc_search = retrieve_query(query)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response



input_text = st.text_input("What would you like to know about this document? ", key="input")
response = retrieve_answers(input_text)

submit = st.button("Ask")

if submit:
    st.subheader("According to the document:")
    st.write(response)
