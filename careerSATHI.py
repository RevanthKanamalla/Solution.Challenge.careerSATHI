import os
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Call the load_dotenv function to load environment variables
load_dotenv()

# Sidebar contents
with st.sidebar:
    st.title('careerSATHI')
    st.markdown('''
    ##
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)
    ''')

# CREATING TABS
tab1, tab2 = st.tabs(["ABOUT", "FILL BELOW FORM"])

# CREATING TAB-1
with tab1:
    st.title('careerSATHI')
    st.caption('NOTE: The server might overload and give an ERROR. If that happens, please close the website and open it again')
    st.title('careerSATHI')
    st.subheader("A GUIDE TO EXCELLENCE")

# CREATING TAB-2
with tab2:
    st.title('FILL THE FORM')

    # Provide the PDF file path directly
    pdf_name = "MCET.pdf"
    current_directory = os.getcwd()
    pdf_path = os.path.join(current_directory, pdf_name)

    # Get the base name (filename) from the path
    pdf_nam = os.path.basename(pdf_path)

    # Use pdf_name instead of pdf.name
    store_name = pdf_nam[:-4]

    VectorStore = None  # Initialize to None

    # Check if the VectorStore already exists
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create VectorStore using LangChain
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Save VectorStore to a pickle file
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    # Get user input
    Rank = st.text_input("Enter the Rank")
    Cast = st.selectbox("Select Category:", ("SC", "OC", "BC", "ST"))
    Course = st.multiselect("Select Course:", ["CSE", "AIML", "AIDS", "ECE", "MECH", "IT"])

    # Define the query variable outside the "Submit" button's scope
    # query = f"My Rank is {Rank}. My category is {Cast}. I am interested in {Course} courses. Suggest me the top 10 colleges according to my requirements"
    query = st.text_input('inout')
    # Check if the "Submit" button is clicked
    if st.button('Submit'):
        st.write(query)  # Display the query

    # Check if the "pdf" button is clicked
    if st.button('pdf'):
        # Check if the query and VectorStore are not None
        if query and VectorStore is not None:
            # Perform similarity search using LangChain
            docs = VectorStore.similarity_search(query=query, k=3)

            # Initialize LangChain components
            llm = OpenAI()
            # chain = load_qa_chain(llm=llm, chain_type="openai-gpt")
            chain = load_qa_chain(llm=llm, chain_type="map_reduce")


            # Run the LangChain pipeline
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            # Display the response
            st.write(response)
