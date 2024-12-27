from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv
import os

load_dotenv()
open_ai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(api_key=open_ai_api_key)


class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, add]


# define our search nodes


def search_web(state: State):
    """Retrieves docs from websearch"""

    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state["question"])
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipidia(state: State):
    """Retrieves docs from wikipidia"""
    # search
    search_docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()
    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def generate_answer(state: State):
    """Node to answer a question"""
    context = state["context"]
    question = state["question"]
    answer_template = """Answer the question {question} using this context:{context}"""
    answer_instruction = answer_template.format(question=question, context=context)
    answer = llm.invoke(
        [SystemMessage(content=answer_instruction)]
        + [HumanMessage(content="Answer the question")]
    )

    return {"answer": answer}


# start building our graph

workflow = StateGraph(State)

# register our nodes

workflow.add_node("search_web", search_web)
workflow.add_node("search_wikipidia", search_wikipidia)
workflow.add_node("generate_answer", generate_answer)

# define our flow
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_wikipidia")
workflow.add_edge("search_web", "generate_answer")
workflow.add_edge("search_wikipidia", "generate_answer")
workflow.add_edge("generate_answer", END)

graph = workflow.compile()

result = graph.invoke({"question": "How were Nvidia's Q2 2024 earnings"})
print(result["answer"].content)
