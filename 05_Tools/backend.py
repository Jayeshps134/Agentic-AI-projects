# -----------------------------------------------------
# imports
# -----------------------------------------------------
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from tools import available_tools
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Tool Chatbot Demo"


# -----------------------------------------------------
# Checkpointer
# -----------------------------------------------------
conn = sqlite3.connect("chatbot_demo.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# -----------------------------------------------------
# LLM
# -----------------------------------------------------
chat_model = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-flash"
)
chat_model_with_tools = chat_model.bind_tools(available_tools)

# -----------------------------------------------------
# State
# -----------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# -----------------------------------------------------
# Nodes
# -----------------------------------------------------
# chat node
def chat_node(state: ChatState):
    response = chat_model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# tool node
tool_node = ToolNode(available_tools)

# -----------------------------------------------------
# Graph
# -----------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

# -----------------------------------------------------
# util func
# -----------------------------------------------------
def retrieve_all_threads():
    unique_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        unique_threads.add(thread_id)
    return list(unique_threads)