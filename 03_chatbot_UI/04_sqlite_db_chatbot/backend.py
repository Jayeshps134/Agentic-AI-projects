from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from langsmith import traceable

# checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Local LLM to generate and optimize post
llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.2,
            "return_full_text": False,
            "max_new_tokens": 1024
        },
    )
chat_model = ChatHuggingFace(llm=llm)


# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# chat node
@traceable(name="chat_node")
def chat_node(state: ChatState):
    model_response = chat_model.invoke(state["messages"])
    return {"messages": [model_response]}


# graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


# # setup
# initial_state = {"messages": [HumanMessage(content="How much time it takes to cook")]}
# config = {"configurable": {"thread_id": '1'}}
#
# # invoke
# final_state = chatbot.invoke(input=initial_state, config=config)
# print(final_state)
#
# # stream
# for msgs, _ in chatbot.stream(input=initial_state, config=config, stream_mode="messages"):
#     print(msgs.content, end=" ", flush=True)

def retrieve_all_threads():
    unique_threads = set()
    for checkpoint in checkpointer.list(None):
        unique_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(unique_threads)