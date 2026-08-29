# imports
import os
from typing import TypedDict, Annotated, List
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "MCP Client"


# MCP Client : server configuration
SERVERS = {
    "Expense Tracker MCP": {
        "transport": "stdio",
        "command": "/Users/jayeshp/.local/bin/uv",
        "args": [
            "run",
            "fastmcp",
            "run",
            "/Users/jayeshp/Desktop/Expense Tracker MCP Server/main.py"
        ]
    }
}


# LLM
chat_model = ChatDeepSeek(api_key=api_key, model="deepseek-v4-flash")


# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# main
async def main():
    # MCP client : tool loading and binding
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    chat_model_with_tools = chat_model.bind_tools(tools)

    # Nodes: chat node & tool node
    async def chat_node(state: ChatState):
        response = chat_model_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    tool_node = ToolNode(tools)

    # async checkpointer
    async with AsyncSqliteSaver.from_conn_string("chatbot.db") as checkpointer:
        # graph
        graph = StateGraph(ChatState)
        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node",tools_condition)
        graph.add_edge("tools","chat_node")
        chatbot = graph.compile(checkpointer=checkpointer)

        # invoke setup
        initial_state = {
            "messages": [HumanMessage(content="create and add 5 expenses for oct-2026 period and then summarize the expenses for oct-2026 period")]
        }
        config = {"configurable": {"thread_id": "3"}}

        response_state = await chatbot.ainvoke(input=initial_state, config=config)
        print("\nAssistant:")
        print(response_state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
