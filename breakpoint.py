from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os

load_dotenv()
# define our llm

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


# define our tool
def multiply(a: int, b: int):
    """multiply two integers a and b
    a:the first integer
    b:the second integer
    """
    return a * b


# register our tool to our llm

llm_with_tools = llm.bind_tools([multiply])

# define our nodes


def assistant(state: MessagesState):
    return {"messages": llm_with_tools.invoke(state["messages"])}


# start building our graph

workflow = StateGraph(MessagesState)

# register our nodes
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode([multiply]))
# build our flow
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")

initial_message = {"messages": "hey, what is 3 multiplied by 4"}

# define our memory
memory = MemorySaver()
graph = workflow.compile(interrupt_before=["tools"], checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

for event in graph.stream(initial_message, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
state = graph.get_state(config)
# print(state.next)
# for m in output["messages"]:
#     m.pretty_print()
if state.next:
    user_approval = input("Do you want to call the tool? (yes/no): ")

    if user_approval.lower() == "yes":
        for event in graph.stream(None, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
    else:
        print("operation cancelled by the user")
