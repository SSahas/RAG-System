import streamlit as st
from src.generation import RAG_Generation

st.title("Document chatbot with RAG")

# Initialize chat history and RAG_Generation instance
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_generator" not in st.session_state:
    st.session_state.rag_generator = RAG_Generation()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        print(st.session_state.messages)
        response = st.session_state.rag_generator.generate_output(prompt)
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
