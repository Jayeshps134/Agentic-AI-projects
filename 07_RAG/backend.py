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
from langchain_core.tools import tool
# RAG components
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import Chroma
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG Client"


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


# models : chat & embedding
chat_model = ChatDeepSeek(api_key=api_key, model="deepseek-v4-flash")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


# RAG components
doc_loader = PyPDFLoader(r"/Users/jayeshp/PycharmProjects/Langgraph/07_RAG/Medical_book.pdf")
docs = doc_loader.load()
txt_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = txt_spliter.split_documents(docs)
chroma_vs = Chroma.from_documents(documents=split_docs, embedding=embedding_model)
retriever = chroma_vs.as_retriever(search_kwargs={"k": 4}, search_type="similarity")

# RAG Tool [Note: this is a demo on how to connect rag. better to create a subgraph]
@tool
def rag_tool(query: str) -> str:
    """
    Retrieves relevant information from the pdf document based on user query.
    Use this tool when the user asks factual or conceptual questions.
    """
    result = retriever.invoke(input=query)
    content = [doc.page_content for doc in result]

    return {"query": query, "content":content}


# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# main
async def main():
    # MCP client : tool loading and binding
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    tools.append(rag_tool)
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
            "messages": [HumanMessage(content="what causes acne and how to treat it")]
        }
        config = {"configurable": {"thread_id": "4"}}

        response_state = await chatbot.ainvoke(input=initial_state, config=config)
        print("\nAssistant:")
        print(response_state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
