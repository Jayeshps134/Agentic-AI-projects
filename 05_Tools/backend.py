from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from tools import available_tools
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Tool Chatbot"

# Checkpointer
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# LLM
chat_model = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-flash"
)
chat_model_with_tools = chat_model.bind_tools(available_tools)

# State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Chat node
def chat_node(state: ChatState):
    response = chat_model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool node
tool_node = ToolNode(available_tools)

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    unique_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        unique_threads.add(thread_id)

    return list(unique_threads)

# # test
# initial_state = {"messages": [HumanMessage("Summarize latest violance incidence in north east india")]}
# config = {"configurable": {"thread_id": 141215}}
#
# for msg, md in chatbot.stream(input=initial_state, config=config,stream_mode="messages"):
#     if isinstance(msg, HumanMessage):
#         print(msg.content, end=" ", flush=True)
#     elif isinstance(msg, ToolMessage):
#         print("Tool: Not printed")
#     elif isinstance(msg, AIMessage):
#         print(msg.content, end=" ", flush=True)

# response_state = chatbot.invoke(input=initial_state, config=config)
#
# for message in response_state["messages"]:
#     if message.type == "human":
#         print(f"User : \n\t{message.content}")
#         print("-"*60)
#     elif message.type == "tool":
#         print(f"Tool called")
#         print("-"*60)
#     elif message.type == "ai":
#         print(f"AI : \n\t{message.content}")
#         print("-"*60)

