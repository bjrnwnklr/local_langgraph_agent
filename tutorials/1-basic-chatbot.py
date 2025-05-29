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
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId

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
    name: str
    birthday: str


# create memory
memory = MemorySaver()

# construct the graph
graph_builder = StateGraph(State)


# define tools
@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        }
    )
    # if the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # explicitely update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # return a Command object in the tool to update the state
    return Command(update=state_update)


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

    user_input = """Can you look up when LangGraph was released?
        When you have the answer, use the human_assistance tool for review."""
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

    human_command = Command(
        resume={
            "name": "LangGraph",
            "birthday": "Jan 17, 2024",
        },
    )

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
