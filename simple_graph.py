from typing_extensions import TypedDict
import random
from typing import Literal

from langgraph.graph import StateGraph, START, END

from IPython.display import Image, display


class State(TypedDict):
    graph_state: str


def node_1(state):
    print("______node_1_______")
    return {"graph_state": state["graph_state"] + "I am"}


def node_2(state):
    print("______node_2_______")
    return {"graph_state": state["graph_state"] + "Happy!"}


def node_3(state):
    print("_____node_3_____")
    return {"graph_state": state["graph_state"] + "Sad!"}


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # get the state to decide what to do based on the state
    user_input = state["graph_state"]

    if random.random() < 0.5:
        return "node_3"
    return "node_2"


# build the graph
builder = StateGraph(State)

# register the nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# build the logic by adding the edges

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)


# compile our graph

graph = builder.compile()

# view the graph

print("nodes", graph.get_graph().nodes)
print("edges", graph.get_graph().edges)
