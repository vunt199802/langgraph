from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
import os


api_key = os.getenv("OPENAI_API_KEY")
# define our llm
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


# create our state with summary property
class State(MessagesState):
    summary: str


# define our nodes


def chatbot(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"summary of the conversation earlier:{summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    return {"messages": llm.invoke(messages)}


def summarize(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"this is the summary of the conversation to date:{summary}\n\n"
            "Extend the summary by taking in to account the new messages"
        )
    else:
        summary_message = "Summarize the above conversation"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages, "summary": response.content}


# define the condition to decide weather to summarize or end


def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"

    return END


# build our graph

builder = StateGraph(State)
# register our nodes
builder.add_node("chatbot", chatbot)
builder.add_node("summarize_conversation", summarize)

# add our logic or edges

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_summarize)
builder.add_edge("summarize_conversation", END)


# add transient(in-memory) checkpointer

memory = MemorySaver()

config = {"configurable": {"thread_id": "1"}}

graph = builder.compile(checkpointer=memory)

input_message = HumanMessage(content="Hi, i am Thomas")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()

input_message = HumanMessage(
    content="i like Nick Bosa, isn't he the highest paid defensive player?"
)
output = graph.invoke({"messages": [input_message]}, config)
for m in output["messages"][-1:]:
    m.pretty_print()

print("summary", graph.get_state(config).values.get("summary", ""))
