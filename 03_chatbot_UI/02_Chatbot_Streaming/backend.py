from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.checkpoint.memory import InMemorySaver


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


# checkpointer
checkpointer = InMemorySaver()


# state
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# chat node
def chat_node(state: ChatState):
    response = chat_model.invoke(input=state["messages"])
    return {"messages": [response]}


# graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


# # test : invoke
# initial_state = {"messages": [HumanMessage(content="What is the recipe of pizza")]}
# response_state = chatbot.invoke(input=initial_state,
#                                 config={"configurable": {"thread_id": "1"}})
#
# print(response_state)


# # test : stream
# initial_state = {"messages": [HumanMessage(content="Give detailed steps to make pizza at home")]}
# config = {"configurable": {"thread_id": "1"}}
# for msg, _ in chatbot.stream(input=initial_state, config=config, stream_mode="messages"):
#     print(msg.content, end=" ", flush=True)
