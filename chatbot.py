import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader, CSVLoader
import time
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config


@st.cache_resource()
def load_embedding_model():
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return bge_embeddings

@st.cache_resource()
def load_llm_model():
    model = "gemini-pro"
    # Define a prompt template instructing the assistant on how to answer investor questions
    model = ChatGoogleGenerativeAI(model=model,google_api_key=config('GOOGLE_API_KEY'),
                             temperature=0.7)

    template = """You are a Chatbot that can answer questions from users based on the Context provided. Ensure the responses are accurate and utilize retrieval from the knowledge base.
    
    Context: {context},
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    return chain

def response_generator(prompt):
    context = st.session_state['retriever'].invoke(prompt)
    print(context)
    result = llm.invoke({'question': prompt, 'context': context})
    
    
    for word in result.content.split():
        yield word + " "
        time.sleep(0.05)
     


bge_embeddings = load_embedding_model()
llm = load_llm_model()

st.title('Chatbot')

file = st.file_uploader('Upload file to chat', type=['csv', 'json', 'pdf'])

if file is not None:
    if st.session_state["is_chat_available"]!=True:
        
        with st.spinner('Wait for it...'):
            with open(file.name, 'wb') as temp:
                    temp.write(file.getvalue())
            if file.type == 'application/pdf':
                loader = PyPDFLoader(file.name)
                
            elif file.type == 'text/csv':
                loader = CSVLoader(file.name, encoding='utf-8')

            elif file.type == 'application/json':
                loader = TextLoader(file.name)

            document = loader.load()
            os.remove(file.name)


            if not isinstance(document, CSVLoader):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
                document = text_splitter.split_documents(document)

            st.session_state["vector_store"] = FAISS.from_documents(documents=document, embedding=bge_embeddings)
            st.session_state["retriever"] = st.session_state["vector_store"].as_retriever(search_type="mmr", search_kwargs={"k": 5})
    st.session_state["is_chat_available"] = True
    
else:
    st.session_state["is_chat_available"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []


if st.session_state["is_chat_available"]:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = st.write_stream(response_generator(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
