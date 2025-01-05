from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.constants import Send
from langgraph.graph import StateGraph, END, START


load_dotenv()
import os


# define our llm model
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)


# define schemas for structured output


class Subjects(BaseModel):
    subjects: list[str]


class BestJoke(BaseModel):
    id: int


# define our overall state


class OverAllState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


# create prompts template

subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}"""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}, select the best one! Return the ID of the best one , starting 0 as the ID for the first joke. jokes:\n\n {jokes}"""

# define our node to generate topics


def generate_topics(state: OverAllState):
    """generates topics"""
    prompt = subjects_prompt.format(topic=state["topic"])
    response = llm.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


def continue_to_jokes(state: OverAllState):
    """generates a joke for each subject parallely by spawning a generate_joke node for each subject"""

    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# define separate states for our generate_joke node


class JokeState(TypedDict):
    subject: str


class Joke(BaseModel):
    joke: str


def generate_joke(state: JokeState):
    """generates a joke"""
    prompt = joke_prompt.format(subject=state["subject"])

    response = llm.with_structured_output(Joke).invoke(prompt)

    return {"jokes": [response.joke]}


def best_joke(state: OverAllState):
    """picks the best joke"""
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)

    response = llm.with_structured_output(BestJoke).invoke(prompt)

    return {"best_selected_joke": state["jokes"][response.id]}


workflow = StateGraph(OverAllState)

workflow.add_node("generate_topics", generate_topics)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("best_joke", best_joke)

workflow.add_edge(START, "generate_topics")
workflow.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
workflow.add_edge("generate_joke", "best_joke")
workflow.add_edge("best_joke", END)

graph = workflow.compile()

for s in graph.stream({"topic": "gym"}):
    print(s)
