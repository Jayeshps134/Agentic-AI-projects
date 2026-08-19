# chat_input, chat_message, session_state

import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []  # message_history = [] # {'role' :"xxx", 'message':"yyy"}

for messages in st.session_state["message_history"]:
    with st.chat_message(messages["role"]):
        st.text(messages["message"])

user_input = st.chat_input("Type here")

if user_input:
    # user
    st.session_state["message_history"].append({"role":"user", "message":user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # assistant
    assistant_state = chatbot.invoke(input={"messages": HumanMessage(content=user_input)},
                   config={"configurable": {"thread_id": "1"}})
    response = assistant_state["messages"][-1].content
    st.session_state["message_history"].append({"role":"assistant", "message":response})
    with st.chat_message("assistant"):
        st.text(response)
