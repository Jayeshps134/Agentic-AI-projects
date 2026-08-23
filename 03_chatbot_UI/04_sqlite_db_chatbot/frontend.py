from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from backend import chatbot, retrieve_all_threads
import streamlit as st
import uuid
from dotenv import load_dotenv
import os
load_dotenv()



# Session states
if "threads" not in st.session_state:
    st.session_state["threads"] = retrieve_all_threads()

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = None


# Util functions
def create_new_chat():
    thread_id = str(uuid.uuid4())
    st.session_state["threads"].append(thread_id)
    st.session_state["current_thread"] = thread_id

def get_chat_history(config):
    state = chatbot.get_state(config=config)
    return state.values.get("messages", [])

def load_chat_history(config):
    for message in get_chat_history(config):
        if message.type == "human":
            with st.chat_message("user"):
                st.write(message.content)
        elif message.type == "tool":
            continue
        elif message.type == "ai" and isinstance(message.content,str) and message.content:
            with st.chat_message("assistant"):
                st.write(message.content)

def stream_assistant_response(initial_state, config):
    for message, metadata in chatbot.stream(input=initial_state, config=config, stream_mode="messages"):
        if isinstance(message, ToolMessage):
            continue

        content = message.content
        if isinstance(content, str) and content:
            yield content


# Sidebar
st.sidebar.title("Chatbot")
if (st.sidebar.button("New Chat") or st.session_state["current_thread"] is None):
    create_new_chat()
    st.rerun()

st.sidebar.subheader("My Conversations")
for thread in st.session_state["threads"]:
    if st.sidebar.button(str(thread), key=f"thread_{thread}"):
        st.session_state["current_thread"] = thread
        st.rerun()

# Config
CONFIG = {
    "configurable": {
        "thread_id": st.session_state["current_thread"]
    }
}

# Load chat history
load_chat_history(CONFIG)

# User input
user_input = st.chat_input("Type here...")
if user_input:
    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write_stream(stream_assistant_response(initial_state,CONFIG))
