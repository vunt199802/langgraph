from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from typing_extensions import TypedDict

# define our node


class MessagesState(TypedDict):
    messages: str


def node_1(state: MessagesState):
    """return the state without modifying it"""
    print("_____node_1______")
    return state


def node_2(state: MessagesState):
    """return the state without modifying it"""
    print("_____node_2_____")

    if len(state["messages"]) > 6:
        raise NodeInterrupt(f"Recieved input that is longer than 6 characters")
    return state


def node_3(state: MessagesState):
    """return the state without modifying it"""
    print("_____node_3_____")
    return state


# construct our graph

workflow = StateGraph(MessagesState)
# register our nodes
workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)
workflow.add_node("node_3", node_3)

# build our flow

workflow.add_edge(START, "node_1")
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", "node_3")
workflow.add_edge("node_3", END)

# define our memory
memory = MemorySaver()

config = {"configurable": {"thread_id": "1"}}
graph = workflow.compile(memory)

initial_message = {"messages": "hi, how are you there"}
for event in graph.stream(initial_message, config, stream_mode="values"):
    print(event)
state = graph.get_state(config)
print(state.next)
print(state.tasks)
