from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import tools_condition, ToolNode

load_dotenv()
import os


# define our state
class State(MessagesState):
    pass


api_key = os.getenv("OPENAI_API_KEY")

# define our chat model
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

system_message = SystemMessage(
    content="You are an arthimethic math expert and your job is to perform arthimetic operations"
)


# define our tools
def add(a: int, b: int) -> int:
    """
    add two numbers and return the result

    args:
        a:the first integer
        b:the second integer
    """

    return a + b


def subtract(a: int, b: int):
    """
    subtracts b from a and returns the result
    args:
        a:the first integer
        b:the second integer
    """
    return a - b


def multiply(a: int, b: int) -> int:
    """
    Multiplies two numbers and returns the result

    args:
        a:the first integer
        b:the second integer
    """
    return a * b


def divide(a: int, b: int) -> float:
    """
    performs integer division, divides a by b

    args:
        a:the first integer
        b:the second integer
    """
    return a / b


# bind our tools to our llm chat model
tools = [add, subtract, multiply, divide]

# now allowing parallel tool calling because arthimetic should be performed in order
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# define our chatbot node


def chatbot(state: State):
    return {"messages": llm_with_tools.invoke([system_message] + state["messages"])}


# start building our graph

builder = StateGraph(State)

# register our node

builder.add_node("chatbot", chatbot)
# our tool calling node must be called tools
builder.add_node("tools", ToolNode([multiply, add, subtract, divide]))

# define our logic or flow or edges

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()

messages = []

messages.append(
    HumanMessage(
        content="i have 5 apples my friend mike gave me double that and then i gave my friend sura 4",
        name="Thomas",
    )
)
output = graph.invoke({"messages": messages})

for m in output["messages"]:
    m.pretty_print()

# print(output["messages"][-1])
