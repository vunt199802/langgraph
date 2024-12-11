from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini")

# define our node


def chat_model_node(state: MessagesState):
    #  trim the messages based on token size before passing it to the llm
    messages = trim_messages(
        state["messages"],
        max_tokens=100,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o-mini"),
        allow_partial=False,
        include_system=True,
    )
    print("trimmed_messages", messages)
    return {"messages": [llm.invoke(messages)]}


# start building our graph

builder = StateGraph(MessagesState)

# register our node
builder.add_node("chat_model", chat_model_node)

# create our flow or logic or edges

builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)

# compile our graph
graph = builder.compile()

messages = [AIMessage(content="Hi", name="bot")]
messages.append(
    SystemMessage(
        content="you are an animal scientist, and your job is to give information about animals"
    )
)
messages.append(HumanMessage(content="Hi", name="Thomas"))
messages.append(
    AIMessage(content="So you said you were researching ocean mammals?", name="bot")
)
messages.append(
    HumanMessage(
        content="Yes, I know about whales. But what others should I learn about?",
        name="Thomas",
    )
)

output = graph.invoke({"messages": messages})
for m in output["messages"]:
    m.pretty_print()
