from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


# define our state with summary included in addition to the messages key
class State(MessagesState):
    summary: str


# define our llm
llm = ChatOpenAI(model="gpt-4o-mini")

# define our nodes


def chat_bot(state: State):
    "interacts with the llm"
    summary = state.get("summary", "")

    if summary:
        system_message = f"summary of the conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message) + state["messages"]]
    else:
        messages = state["messages"]

    return {"messages": llm.invoke(messages)}


def summarize(state: State):
    "summarizes the conversation history and also removes old messages"

    summary = state.get("summary", "")

    # if we have a summary we tell the llm to update the summary based on new messages
    if summary:
        summary_message = (
            f"this is summary of the conversation to date: {summary} \n\n"
            "Extend the summary by taking into account the new messages above"
        )
    # if we don't have a summary then we just summarize our conversations so far
    else:
        summary_message = "Create a summary of the conversation above"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # delete all but the 2 most recent messages from the conversation history
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# a function to determine whether to summarize or end the conversation


def should_continue(state: State):
    "Return the next node to excute based on the length of messages"
    messages = state.get("messages", "")

    if len(messages) > 6:
        return "summarize_conversation"

    return END


# define our memory saver

memory = MemorySaver()

config = {"configurable": {"thread_id": "1"}}

# build our graph
builder = StateGraph(State)

# register our nodes
builder.add_node("summarize_conversation", summarize)
builder.add_node("chat_bot", chat_bot)

# build the flow or logic or edges of our graph

builder.add_edge(START, "chat_bot")
builder.add_conditional_edges("chat_bot", should_continue)
builder.add_edge("summarize_conversation", END)

graph = builder.compile(checkpointer=memory)

input_message = HumanMessage(content="Hey, I'm Thomas")

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
