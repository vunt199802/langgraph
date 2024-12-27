from IPython.display import Image, display
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError
from typing import Annotated
import operator


class State(TypedDict):
    state: Annotated[list, operator.add]


# define our node
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self.value = node_secret

    def __call__(self, state: State):
        print(f"Adding {self.value} to {state['state']}")
        return {"state": [self.value]}


# start building our graph

workflow = StateGraph(State)

# register our nodes

workflow.add_node("a", ReturnNodeValue("I am A"))
workflow.add_node("b", ReturnNodeValue("I am B"))
workflow.add_node("b2", ReturnNodeValue("I'm B2"))
workflow.add_node("c", ReturnNodeValue("I am C"))
workflow.add_node("d", ReturnNodeValue("I am D"))

workflow.add_edge(START, "a")
workflow.add_edge("a", "b")
workflow.add_edge("a", "c")
workflow.add_edge("b", "b2")
workflow.add_edge("b2", "d")
workflow.add_edge("c", "d")
workflow.add_edge("d", END)

graph = workflow.compile()


try:
    graph.invoke({"state": []})
except InvalidUpdateError as e:
    print(f"an error occured: {e}")
