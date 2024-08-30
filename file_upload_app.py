# Import Libraries 
import openai
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.llms import OpenAI
import os 
import streamlit as st 
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

# Initialise Streamlit App
st.set_page_config(page_title="PDF Reader")
st.header("PDF Reader")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file in temp directory
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name=uploaded_file.name

    # Read Input PDF 
    def read_doc(file):
        file_loader = PyPDFLoader(file)
        document = file_loader.load()
        return document

    doc = read_doc(temp_file)

    # # Display PDF content (optional)
    # for page in doc:
    #     st.write(page.page_content)

    # Divide documents into chunks 
    def chunk_data(docs, chunk_size=800, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(docs)
        return docs

    doc_chunk = chunk_data(docs=doc)

    # Embed Document Chunks
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

    # Initialise Pinecone database
    pinecone_api_key = os.environ["PINECONE_API_KEY"]

    vector_db = Pinecone.from_documents(
            documents=doc_chunk, 
            embedding=OpenAIEmbeddings(), 
            index_name="llm-app",
    )

    # Retrieve result with cosine similarity 
    def retrieve_query(query, k=6):
        matching_result = vector_db.similarity_search(query, k=k)
        return matching_result

    # Initialising LLM model 
    # prompt_template = """
    #                     Please provide a detailed and thorough response to the following question, 
    #                     with as much relevant information as possible.

    #                     Question: {question}

    #                     Answer:
    #                     """
    
    # prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.6)

    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    chain = load_qa_chain(llm, chain_type='refine')

    # Retrieving answers 
    def retrieve_answers(query):
        doc_search = retrieve_query(query)
        response = chain.run(input_documents=doc_search, question=query)
        return response

    # User Input and Response
    input_text = st.text_input("What would you like to know about this document?", key="input")
    submit = st.button("Ask")

    if submit:
        with st.spinner("Processing..."):
            response = retrieve_answers(input_text)
        st.subheader("According to the document:")
        st.write(response)

    # Download Response Button
        if response:
            st.download_button("Download Response", response, file_name="response.txt")
