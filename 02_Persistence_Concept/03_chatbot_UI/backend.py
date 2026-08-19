from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.checkpoint.memory import InMemorySaver


# Local LLM to generate and optimize post
llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.2,
            "return_full_text":False,
        },
    )
chat_model = ChatHuggingFace(llm=llm)

# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# chat node func
def chat_node(state: ChatState):
    response = chat_model.invoke(state["messages"])
    return {"messages": [response]}


# graph
checkpointer = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


# # test
# while True:
#     msg = input("Q: ")
#     if msg.strip().lower() in ["q"]:
#         break
#
#     config = {'configurable': {"thread_id": "1"}} # store/load all conversation related to this thread
#
#     response = chatbot.invoke(input={"messages": [msg]}, config=config)
#     print("AI :", response["messages"][-1].content)