#!/usr/bin/env python3
"""
Local LangGraph starter.

✓ Runs fully offline (Ollama + Chroma)
✓ Can be inspected live via `langgraph dev`
"""

import os
from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama  # local LLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = Path("./data/chroma")  # persistent vector store
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
TEMPERATURE = float(os.getenv("TEMP", 0.7))


def make_llm():
    """Connect to the running Ollama daemon."""
    return ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)


def make_retriever():
    """Create (or reuse) a Chroma collection for long-term memory."""
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    vectordb = Chroma(
        persist_directory=str(DATA_PATH),
        embedding_function=embeddings,
        collection_name="agent_knowledge",
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def build_agent():
    llm = make_llm()
    agent = create_react_agent(
        model=llm,
        tools=[],  # add tools later (e.g. browser, file-ops)
        prompt="You are a helpful research assistant.",
    )
    return agent


if __name__ == "__main__":
    agent = build_agent()
    print("💡  Type 'quit' to exit.")
    while True:
        user = input("\n👤  ")
        if user.lower() in {"quit", "exit"}:
            break

        result = agent.invoke({"messages": [{"role": "user", "content": user}]})

        # result is an AddableValuesDict; pull out the text
        response_text = None
        messages = result.get("messages")
        if messages:
            # messages is a list with 2 entries, "HumanMessage" and "AIMessage"
            last = messages[-1]
            # message objects have a .content attribute
            response_text = getattr(last, "content", str(last))
        if response_text is None:
            # fallback to full dict repr
            response_text = str(result)

        print("🤖", response_text)
