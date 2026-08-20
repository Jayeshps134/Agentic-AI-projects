import streamlit as st
from langchain_core.messages import HumanMessage
from backend import chatbot

# session state for streamlit
if "message_history" not in st.session_state:
    st.session_state["message_history"] = [] # {"role": "user", "content": "hi"}


# load all messages
for messages in st.session_state["message_history"]:
    with st.chat_message(messages["role"]):
        st.write(messages["content"])


# conversations
user_input = st.chat_input("Type here")
if user_input:
    # user
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # assistant
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable" : {"thread_id": "1"}}

    with st.chat_message("assistant"):
        ai_response = st.write_stream(msg.content for msg, _ in chatbot.stream(input=initial_state, config=config, stream_mode="messages"))

    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
