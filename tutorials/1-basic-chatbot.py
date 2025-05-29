import os
import json
from pathlib import Path
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage

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


llm = init_chat_model("openai:gpt-4.1-mini")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    too_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(state: State):
    """Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    print(f"Invoking route_tools. {state=}")
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in the input state to tool_edge: {state}")
    # check if any tool_calls in the ai_message, then route to tools node
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("route_tools returning 'tools'.")
        return "tools"
    # route to the end otherwise
    return END


# construct the graph
graph_builder = StateGraph(State)

# define tools
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# add a node for the tools and add to graph
# tool_node = BasicToolNode(tools=[tool])
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# The 'route_tools' function returns "tools" if the chatbot asks to use a tool, and 'END' if
# it is fine directly responding. This conditional routing defines the main agent loop.
# graph_builder.add_conditional_edges(
#     "chatbot",
#     route_tools,
#     # dictionary tells the graph to interpret the condition functions
#     # as specific nodes. You can use this if you want to provide a specific name
#     # for 'tools', e.g. "tools": "my_tools"
#     {"tools": "tools", END: END},
# )
# any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# add starting edge for entry
# this will go from START to the chatbot
graph_builder.add_edge(START, "chatbot")
# compile the graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """Process user input messages and return the answer."""
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


def main():

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye my friend!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about Iron Maiden?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


if __name__ == "__main__":
    main()
