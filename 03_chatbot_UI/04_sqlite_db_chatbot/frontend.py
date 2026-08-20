from langchain_core.messages import HumanMessage
from backend import chatbot, retrieve_all_threads
import streamlit as st
import uuid

# -------------------------------- session states --------------------------------
if "threads" not in st.session_state:
    st.session_state["threads"] = retrieve_all_threads()

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = None

# -------------------------------- util functions --------------------------------
def create_new_chat():
    thread_id = str(uuid.uuid4())
    st.session_state["threads"].append(thread_id)
    st.session_state["current_thread"] = thread_id

def get_chat_history(config):
    state = chatbot.get_state(config=config)
    return state.values.get("messages", [])

def load_chat_history(config):
    for message in get_chat_history(config):
        # user
        if message.type == "human":
            with st.chat_message("user"):
                st.text(message.content)
        # assistant
        elif message.type == "ai":
            with st.chat_message("assistant"):
                st.text(message.content)

# -------------------------------- side bar --------------------------------
st.sidebar.title("Chatbot")
if (st.sidebar.button("New Chat")) or (st.session_state["current_thread"] is None):
    create_new_chat()
    st.rerun()

st.sidebar.subheader("My Conversations")
for thread in st.session_state["threads"]:
    if st.sidebar.button(str(thread)):
        st.session_state["current_thread"] = thread
        st.rerun()

# -------------------------------- main UI --------------------------------
CONFIG = {"configurable": {"thread_id": st.session_state["current_thread"]}}

# load chat history
load_chat_history(CONFIG)

# User input & conversation
user_input = st.chat_input("type here")

if user_input:
    # user
    initial_state = {"messages": HumanMessage(user_input)}
    with st.chat_message("user"):
        st.text(user_input)

    # assistant
    with st.chat_message("assistant"):
        st.write_stream(msg.content for msg, _ in chatbot.stream(input=initial_state, config=CONFIG, stream_mode="messages"))
