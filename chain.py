from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


# define our message state using the default MessagesState by langgraph which contains the key messages with the reducer add_messages
class MessagesState(MessagesState):
    pass


# define our external tool


def multiply(a: int, b: int) -> int:
    """Multiply a and b
    Args:
        a:first int
        b: second int

    """

    return a * b


def add(a: int, b: int) -> int:
    """Add a and b
    Args:
        a:first int
        b:second int

    """
    return a + b


def subtract(a: int, b: int) -> int:
    """subtract b from a
    Args:
        a:first int
        b:second int

    """
    return a - b


# define a system message or a prompt
sys_message = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)
#  bind our external tool to our llm


llm_with_tools = llm.bind_tools([multiply, subtract, add])


# tool_call = llm_with_tools.invoke(
#     [HumanMessage(content=f"how much is 2 times 3", name="Thomas")]
# )

# print(tool_call.additional_kwargs["tool_calls"])


# define our tool calling node
def tool_calling_node(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}


# define our Math assistant chatbot node


def math_assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_message] + state["messages"])]}


# build the graph

builder = StateGraph(MessagesState)

# register our node
builder.add_node("math_assistant", tool_calling_node)
builder.add_node("tools", ToolNode([multiply, add, subtract]))
# add our flow logic or edges
builder.add_edge(START, "math_assistant")

# Use in the conditional_edge to route to the ToolNode if the last message
# has tool calls. Otherwise, route to the end.
builder.add_conditional_edges("math_assistant", tools_condition)
builder.add_edge("tools", "math_assistant")

# compile our graph
graph = builder.compile()


# print("nodes", graph.get_graph().nodes)
# print("edges", graph.get_graph().edges)
# print("graph_json", graph.get_graph().to_json())
initial_message = HumanMessage(
    content="Hi, Add 3 and 4. Multiply the output by 2. Divide the output by 5",
    name="Thomas",
)
result = graph.invoke({"messages": [initial_message]})
for message in result["messages"]:
    message.pretty_print()
