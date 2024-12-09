from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START


# input schema
class InputState(TypedDict):
    question: str


# output schema
class OutPutState(TypedDict):
    answer: str


# overall internal graph schema that contains all the information all the nodes in the graph need
class OverAllState(TypedDict):
    answer: str
    question: str
    notes: str


# define our thinking node
def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his name is Lance"}


# define our answer node
def answer_node(state: OverAllState) -> OutPutState:
    return {"answer": "bye Lance"}


# create our graph builder with the state, including input AND output schema
builder = StateGraph(OverAllState, input=InputState, output=OutPutState)

# register our nodes
builder.add_node("thinking_node", thinking_node)
builder.add_node("answer_node", answer_node)

# build the logic or flow or edges of our graph

builder.add_edge(START, "thinking_node")
builder.add_edge("thinking_node", "answer_node")
builder.add_edge("answer_node", END)

graph = builder.compile()


result = graph.invoke({"question": "hi"})

print(result)
