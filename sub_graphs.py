from langgraph.graph import StateGraph, START, END
from operator import add
from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
import time

load_dotenv()

"""

create a multi agent calculator 1 agent for each operation and each agent will be a subgraph, 
the graph will take two numbers and perform the 4 operations on each of them


"""

# Define our overall states


class CalculatorState(TypedDict):
    first_number: float
    second_number: float
    addition_result: float
    subtraction_result: float
    multiplication_result: float
    division_result: float
    error: str
    summary: str


# define our addition subgraph state
class AdditionState(TypedDict):
    first_number: float
    second_number: float
    addition_result: float


class AdditionOutPutState(TypedDict):
    addition_result: float


# define our subtraction subgraph state
class SubtractionState(TypedDict):
    first_number: float
    second_number: float
    subtraction_result: float


class SubtractionOutPutResult(TypedDict):
    subtraction_result: float


# define our multiplication subgraph state
class MultiplicationState(TypedDict):
    first_number: float
    second_number: float
    multiplication_result: float


class MultiplicationOutPutState(TypedDict):
    multiplication_result: float


# define our division subgraph state
class DivisionState(TypedDict):
    first_number: float
    second_number: float
    division_result: float
    error: str


class DivisionOutPutState(TypedDict):
    division_result: float
    error: str


"""################################create our addition subgraph#######################################"""

# define our addition node


def add(state: AdditionState) -> float:
    """add two numbers"""
    time.sleep(2)
    first_number = state["first_number"]
    second_number = state["second_number"]
    result = first_number + second_number
    return {"addition_result": result}


add_workflow = StateGraph(
    input=AdditionState, output=AdditionOutPutState, state_schema=AdditionState
)
add_workflow.add_node("add", add)
add_workflow.add_edge(START, "add")
add_workflow.add_edge("add", END)


"""################################create our subtraction subgraph#######################################"""

# add our subtraction node


def subtract(state: SubtractionState) -> float:
    """subtracts the smaller value from the larger value and return the difference"""
    time.sleep(2)
    first_number = state["first_number"]
    second_number = state["second_number"]
    if first_number >= second_number:
        result = first_number - second_number
    else:
        result = second_number - first_number

    return {"subtraction_result": result}


subtract_workflow = StateGraph(
    input=SubtractionState,
    output=SubtractionOutPutResult,
    state_schema=SubtractionState,
)
subtract_workflow.add_node("subtract", subtract)
subtract_workflow.add_edge(START, "subtract")
subtract_workflow.add_edge("subtract", END)


"""################################create our multiplication subgraph#######################################"""

# define our multiplication node


def multiply(state: MultiplicationState) -> float:
    time.sleep(2)
    """multiplies two numbers and returns the result"""
    first_number = state["first_number"]
    second_number = state["second_number"]
    result = first_number * second_number
    return {"multiplication_result": result}


multiply_workflow = StateGraph(
    input=MultiplicationState,
    output=MultiplicationOutPutState,
    state_schema=MultiplicationState,
)
multiply_workflow.add_node("multiply", multiply)
multiply_workflow.add_edge(START, "multiply")
multiply_workflow.add_edge("multiply", END)

"""################################create our division subgraph#######################################"""

# define our division node


def divide(state: DivisionState) -> float:
    """divides the first number by the second number and returns the result"""
    time.sleep(2)
    numerator = state["first_number"]
    denominator = state["second_number"]
    if denominator == 0:
        return {"error": "Division by zero error"}
    result = numerator / denominator
    return {"division_result": result}


division_workflow = StateGraph(
    input=DivisionState, output=DivisionOutPutState, state_schema=DivisionState
)
division_workflow.add_node("divide", divide)
division_workflow.add_edge(START, "divide")
division_workflow.add_edge("divide", END)


"""################################ create our calculator graph #######################################"""


# define our llm

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")


# define our summary node
def summerize(state: CalculatorState):
    """summerize the results of all the operations"""

    return {
        "summary": f"addition result is {state['addition_result']}, subtraction result is {state['subtraction_result']}, multiplication result is {state['multiplication_result']}, division result is {state['division_result']}"
    }


calculator_workflow = StateGraph(CalculatorState)
calculator_workflow.add_node("summarization", summerize)
calculator_workflow.add_node("addition", add_workflow.compile())
calculator_workflow.add_node("subtraction", subtract_workflow.compile())
calculator_workflow.add_node("multiplication", multiply_workflow.compile())
calculator_workflow.add_node("division", division_workflow.compile())


calculator_workflow.add_edge(START, "addition")
calculator_workflow.add_edge(START, "subtraction")
calculator_workflow.add_edge(START, "multiplication")
calculator_workflow.add_edge(START, "division")
calculator_workflow.add_edge("addition", "summarization")
calculator_workflow.add_edge("subtraction", "summarization")
calculator_workflow.add_edge("multiplication", "summarization")
calculator_workflow.add_edge("division", "summarization")
calculator_workflow.add_edge("summarization", END)

# calculator_workflow.add_edge(START, "addition")
# calculator_workflow.add_edge("addition", "subtraction")
# calculator_workflow.add_edge("subtraction", "multiplication")
# calculator_workflow.add_edge("multiplication", "division")
# calculator_workflow.add_edge("division", "summarization")
# calculator_workflow.add_edge("summarization", END)

calculator_graph = calculator_workflow.compile()

# image_data = calculator_graph.get_graph(xray=1).draw_mermaid_png()

# with open("image2.png", "wb") as file:
#     file.write(image_data)

start_time = time.time()

result = calculator_graph.invoke({"first_number": 10, "second_number": 4})
print(result["summary"])
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")
