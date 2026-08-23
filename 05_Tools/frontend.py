# -----------------------------------------------------
# imports
# -----------------------------------------------------
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from backend import chatbot, retrieve_all_threads
import streamlit as st
import uuid
from dotenv import load_dotenv
load_dotenv()


# -----------------------------------------------------
# Session state
# -----------------------------------------------------
if "threads" not in st.session_state:
    st.session_state["threads"] = retrieve_all_threads()

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = None


# -----------------------------------------------------
# Utils
# -----------------------------------------------------
def create_new_chat():
    thread_id = str(uuid.uuid4())
    st.session_state["threads"].append(thread_id)
    st.session_state["current_thread"] = thread_id

def get_chat_history(config):
    state = chatbot.get_state(config=config)
    return state.values.get("messages", [])

def get_message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = []
        for item in content:
            if isinstance(item, str):
                text.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text.append(item.get("text", ""))
        return "".join(text)
    return str(content)

def load_chat_history(config):
    for message in get_chat_history(config):
        if isinstance(message, HumanMessage) or message.type == "human":
            with st.chat_message("user"):
                st.write(get_message_text(message.content))
        elif isinstance(message, AIMessage) or message.type == "ai":
            if message.content:
                with st.chat_message("assistant"):
                    st.write(get_message_text(message.content))
        elif isinstance(message, ToolMessage) or message.type == "tool":
            with st.chat_message("tool"):
                st.write(get_message_text(message.content))


# -----------------------------------------------------
# Side bar
# -----------------------------------------------------
st.sidebar.title("Chatbot")
if (st.sidebar.button("New Chat")) or (st.session_state["current_thread"] is None):
    create_new_chat()
    st.rerun()

st.sidebar.subheader("My Conversations")
for thread in st.session_state["threads"]:
    if st.sidebar.button(str(thread), key=f"thread_{thread}"):
        st.session_state["current_thread"] = thread
        st.rerun()


# -----------------------------------------------------
# Main UI
# -----------------------------------------------------
CONFIG = {"configurable": {"thread_id": st.session_state["current_thread"]}}
load_chat_history(CONFIG)

user_input = st.chat_input("type here")
if user_input:
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    # user
    with st.chat_message("user"):
        st.write(user_input)

    # assistant
    with st.chat_message("assistant"):
        def generate_response():
            for msg, _ in chatbot.stream(input=initial_state, config=CONFIG, stream_mode="messages"):
                if isinstance(msg, AIMessage) and msg.content:
                    content = get_message_text(msg.content)
                    if content:
                        yield content
        st.write_stream(generate_response())
    st.rerun()