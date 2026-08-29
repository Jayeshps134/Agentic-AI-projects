# imports
import os
from typing import TypedDict, Annotated, List
import asyncio
import yfinance as yf
from datetime import date, timedelta
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt, Command


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "HITL Client"


# models : chat & embedding
chat_model = ChatDeepSeek(api_key=api_key, model="deepseek-v4-flash")


# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]



@tool
def get_stock_price(ticker: str, target_date: date) -> str:
    """
    Fetch the closing stock price for a ticker on a particular date.
    Args:
        ticker: Stock ticker symbol, e.g. AAPL or RELIANCE.NS.
        target_date: Date for which the stock price is required format is YYYY-MM-DD.
    """
    try:
        stock = yf.Ticker(ticker)
        start_date = target_date
        end_date = target_date + timedelta(days=1)

        history = stock.history(
            start=start_date,
            end=end_date
        )
        if history.empty:
            return (
                f"No stock price data found for {ticker.upper()} "
                f"on {target_date}."
            )
        closing_price = history["Close"].iloc[0]

        return (
            f"{ticker.upper()} closing price on "
            f"{target_date}: {closing_price:.2f}"
        )

    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


@tool
def purchase_stock(ticker: str, quantity: int, price: float) -> str:
    """
    Purchase stock for a particular ticker.

    Args:
        ticker: Stock ticker symbol, e.g. AAPL or RELIANCE.NS.
        quantity: Number of shares to purchase.
        price: Price per share at which the simulated purchase is made.
    """
    if quantity <= 0:
        return "Quantity must be greater than 0."
    if price <= 0:
        return "Price must be greater than 0."

    # HITL code: confirm user wants to make purchase
    decision = interrupt(f"Approve buying {quantity} shares of {ticker} at ${price} per share? Yes/No ")

    if decision == "No":
        return f"Purchase of {ticker} stocks declined by user"

    total_value = quantity * price
    return (
        f"Dummy purchase successful!\n"
        f"Ticker: {ticker.upper()}\n"
        f"Quantity: {quantity}\n"
        f"Price per share: {price:.2f}\n"
        f"Total value: {total_value:.2f}"
    )
tools = [get_stock_price, purchase_stock]
chat_model_with_tools = chat_model.bind_tools(tools)


# main
async def main():
    # Nodes: chat node & tool node
    async def chat_node(state: ChatState):
        response = await chat_model_with_tools.ainvoke(state["messages"])
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
        while True:
            user_input = input('User: ')
            if user_input == "quit":
                break

            initial_state = {"messages": [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": "20"}}

            response_state = await chatbot.ainvoke(input=initial_state, config=config)

            # HITL
            interrupts = response_state.get("__interrupt__", [])
            if interrupts:
                print(f"HITL: {interrupts[0].value}")
                decision = input("Your decision: ")

                # resume chatbot with command
                response_state = await  chatbot.ainvoke(input=Command(resume=decision), config=config)

            print(f"\nAssistant: {response_state["messages"][-1].content}")
            print("-"*60)


if __name__ == "__main__":
    asyncio.run(main())
