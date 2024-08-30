This is a Streamlit application that allows users to upload a PDF document and ask questions about its contents using a language model. 
The code integrates OpenAI's GPT model, LangChain, and Pinecone for document processing. The code is broken down below:

### Import Libraries and Set up Environment
The necessary libraries were first imported.
'load_dotenv()' was used to set up the .env files 

### Initialise Webpage(Streamlit)
The page header and title were initialised using streamlit's 'header' and 'set_page_config methods'.
A file uploader widget was created using 'st.file_uploader' function.

### Reading the Uploaded file
The uploaded file was then saved as a temp file so that PyPDFLoader can read it. 
PyPDfLoader's read function saves the uploaded file as an object



