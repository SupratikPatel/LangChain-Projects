import streamlit as st
import requests

# Initialize session state variables
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'current_input' not in st.session_state:
    st.session_state.current_input = None
if 'response' not in st.session_state:
    st.session_state.response = ""

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/Online_model/invoke",
                             json={'input': {'topic': input_text}})
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/Offline_model/invoke",
                             json={'input': {'topic': input_text}})
    return response.json()['output']

def handle_perplexity():
    if not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.current_input = 'input_text'
        st.session_state.response = get_openai_response(st.session_state.input_text)
        st.session_state.input_text = ""  # Clear the input field
        st.session_state.is_processing = False
        st.session_state.current_input = None

def handle_llama3():
    if not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.current_input = 'input_text1'
        st.session_state.response = get_ollama_response(st.session_state.input_text1)
        st.session_state.input_text1 = ""  # Clear the input field
        st.session_state.is_processing = False
        st.session_state.current_input = None

st.title('Langchain FastAPI deploy LLMs as API')

# Text input for Perplexity with callback
input_text = st.text_input("Ask PPLX", key='input_text', on_change=handle_perplexity)

# Text input for LLAMA3 with callback
input_text1 = st.text_input("Ask LLAMA3", key='input_text1', on_change=handle_llama3)

# Placeholder for the response
response_placeholder = st.empty()

# Display the response
if st.session_state.response:
    response_placeholder.write(st.session_state.response)

# User feedback if a request is already being processed
if st.session_state.is_processing:
    st.write("Please wait for the current request to complete.")