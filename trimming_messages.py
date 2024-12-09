from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.graph.message import RemoveMessage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# instantiate our llm model
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# define our nodes


def filter_messages(state: MessagesState):
    # delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}


def chat_model_node(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


# register our nodes
builder = StateGraph(MessagesState)
builder.add_node("filter_messages", filter_messages)
builder.add_node("chat_model", chat_model_node)

# build our logic flow or our edges
builder.add_edge(START, "filter_messages")
builder.add_edge("filter_messages", "chat_model")
builder.add_edge("chat_model", END)

graph = builder.compile()

messages = [AIMessage(content="Hi", name="bot", id="1")]
messages.append(HumanMessage(content="hi", name="Thomas", id="2"))
messages.append(
    AIMessage(
        content="So you said you were researching ocean mammals?", name="bot", id="3"
    )
)
messages.append(
    HumanMessage(
        content="Yes, I know about whales. But what others should I learn about?",
        name="Thomas",
        id="4",
    )
)
output = graph.invoke({"messages": messages})

for m in output["messages"]:
    m.pretty_print()
