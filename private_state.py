from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START


class PrivateState(TypedDict):
    baz: int


class OverAllState(TypedDict):
    foo: int


def node_1(state: OverAllState) -> PrivateState:
    return {"baz": state["foo"] + 1}


def node_2(state: PrivateState) -> OverAllState:
    return {"foo": state["baz"] + 1}


builder = StateGraph(OverAllState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()

result = graph.invoke({"foo": 1})
print(result)
