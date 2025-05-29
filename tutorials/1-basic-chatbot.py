import os
from pathlib import Path
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_tavily import TavilySearch

# load API keys
# 1. Compute project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".local.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .local.env")
# load TAVILY search engine key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY in .local.env")

config = {"configurable": {"thread_id": "1"}}
llm = init_chat_model("openai:gpt-4.1-mini")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# create memory
memory = MemorySaver()

# construct the graph
graph_builder = StateGraph(State)


# define tools
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


search_tool = TavilySearch(max_results=2)
tools = [search_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


# define chatbot and add to graph
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

# add a node for the tools and add as conditional edge to graph
# using the prebuilt ToolNode and tools_condition,
# this will route from the chatbot if the returned ai_message
# contains an attribute "tool_calls", which is the universal
# standard in LLMs to show that a tool needs to be called.
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# compile the graph
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    """Process user input messages and return the answer."""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    ):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


def main():

    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print(snapshot)
    print(snapshot.next)

    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # while True:
    #     try:
    #         user_input = input("User: ")
    #         if user_input.lower() in ["quit", "exit", "q"]:
    #             print("Goodbye my friend!")
    #             break
    #         stream_graph_updates(user_input)
    #         # inspect state
    #         # snapshot = graph.get_state(config)
    #         # print(snapshot)
    #     except:
    #         # fallback if input() is not available
    #         # user_input = "What do you know about Iron Maiden?"
    #         # print("User: " + user_input)
    #         # stream_graph_updates(user_input)
    #         break


if __name__ == "__main__":
    main()
