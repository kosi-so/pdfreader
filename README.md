This is a Streamlit application that allows users to upload a PDF document and ask questions about its contents using a language model. 
The code integrates OpenAI's GPT model, LangChain, and Pinecone for document processing. The code is broken down below:

### 1) Import Libraries and Set up Environment
The necessary libraries were first imported.
'load_dotenv()' was used to set up the .env files. 

### 2) Initialise Webpage(Streamlit)
The page header and title were initialised using streamlit's 'header' and 'set_page_config methods'.
A file uploader widget was created using 'st.file_uploader' function.

### 3) Reading the Uploaded file
The uploaded file was then saved as a temp file so that PyPDFLoader can read it. 
PyPDfLoader's read function saves the uploaded file as an object.

### 4) Dividing the Document into Chunks and Embedding them
RecursiveCharacterTextSPlitter was then used to divide the saved document into chunks for Embedding.
Document chunks were embedded with OpenAIEmbeddings 

### 5) Initializing Pinecone Database
The embedded chunks were stored using a pinecone database. 

### 6) Query retrieval
"retrieve_query" function searches for chunks within the database that are most similar to the user's query. 

### 7)  Initialise LLM Model and QA chain
A gpt 3.5 turbo-instruct model was initialised with a temperature of 0.6, and QA chain was initialised with 'refine' chain type 

### 8) Answer retrieval 
The 'retrieve answer' function takes a question from the user and returns an answer to the input. 
It takes the question as a query and uses the retrieve_query function to select document chunks that are most similar to it.
The document chunk and query are then passed into the LLM model, which generates the answer to the query as the output. 

### 9) User Input and Response
Using the st.text_input method, the webpage asks users for their question and returns the output of the 'retrieve_answer' function.









